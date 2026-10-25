"""Tests for harmonize, image_stats, quality, dedupe, transform and split."""

import random

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from scripts.common.config import load_yaml, resolve_path
from scripts.preprocess import dedupe, harmonize, image_stats, quality, split, transform

# --- helpers --------------------------------------------------------------------


def textured_image(path, seed=0, size=(300, 260), brightness=1.0, blur=0):
    """Random-noise RGB image: sharp, mid-grey on average, unique per seed."""
    rng = np.random.default_rng(seed)
    pixels = (rng.integers(40, 216, size=(size[1], size[0], 3)) * brightness)
    img = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


@pytest.fixture
def configs(params):
    return (load_yaml(resolve_path(params, "harmonization")),
            load_yaml(resolve_path(params, "urgency_mapping")))


def write_raw_datasets(raw):
    """Tiny but realistic copies of all four metadata files plus images."""
    (raw / "isic").mkdir(parents=True)
    pd.DataFrame({
        "isic_id": ["ISIC_0000001", "ISIC_0024306", "ISIC_0000003"],
        "diagnosis_1": ["Malignant", "Benign", "Indeterminate"],
        "diagnosis_2": ["Malignant melanocytic proliferations (Melanoma)",
                        "Benign melanocytic proliferations", "Something new"],
        "diagnosis_3": ["Melanoma, NOS", "Nevus", "Unseen label"],
        "anatom_site_1": ["Trunk", "Lower extremity", "Trunk"],
        "anatom_site_2": ["Posterior trunk", "Foot", None],
        "sex": ["female", "male", None], "age_approx": [55, 130, 40],
        "copyright_license": ["CC-0"] * 3,
    }).to_csv(raw / "isic/metadata.csv", index=False)
    textured_image(raw / "isic/images/ISIC_0000001.jpg", seed=1)
    textured_image(raw / "isic/images/ISIC_0024306.jpg", seed=2)
    textured_image(raw / "isic/images/ISIC_0000003.jpg", seed=3)

    (raw / "ham10000").mkdir(parents=True)
    pd.DataFrame({
        "lesion_id": ["HAM_1", "HAM_1", "HAM_2"],
        "image_id": ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308"],
        "dx": ["nv", "nv", "akiec"], "dx_type": ["histo"] * 3,
        "age": [45, 45, None], "sex": ["male", "male", "unknown"],
        "localization": ["foot", "foot", "face"], "dataset": ["x"] * 3,
    }).to_csv(raw / "ham10000/metadata.csv", index=False)
    textured_image(raw / "ham10000/images/ISIC_0024306.jpg", seed=2)  # same photo as ISIC
    textured_image(raw / "ham10000/images/ISIC_0024307.jpg", seed=4)
    # ISIC_0024308 image deliberately missing

    (raw / "pad_ufes_20").mkdir(parents=True)
    pd.DataFrame({
        "patient_id": ["PAT_1", "PAT_1"], "lesion_id": [10, 11],
        "img_id": ["PAT_1_10_1.png", "PAT_1_11_2.png"],
        "diagnostic": ["BCC", "MEL"], "age": [70, 70], "gender": ["FEMALE", None],
        "region": ["NOSE", "BACK"], "fitspatrick": [2.0, None],
        "itch": ["True", "UNK"], "grew": ["False", "True"], "hurt": ["False", "False"],
        "changed": ["UNK", "True"], "bleed": ["True", "False"],
        "elevation": ["True", "False"],
    }).to_csv(raw / "pad_ufes_20/metadata.csv", index=False)
    textured_image(raw / "pad_ufes_20/images/PAT_1_10_1.png", seed=5)
    textured_image(raw / "pad_ufes_20/images/PAT_1_11_2.png", seed=6)

    (raw / "fitzpatrick17k").mkdir(parents=True)
    pd.DataFrame({
        "md5hash": ["aaa", "bbb"], "fitzpatrick_scale": [6, -1],
        "label": ["kaposi sarcoma", "psoriasis"],
        "three_partition_label": ["malignant", "non-neoplastic"], "url": ["u1", "u2"],
    }).to_csv(raw / "fitzpatrick17k/metadata.csv", index=False)
    textured_image(raw / "fitzpatrick17k/images/aaa.jpg", seed=7)
    textured_image(raw / "fitzpatrick17k/images/bbb.jpg", seed=8)


# --- harmonize --------------------------------------------------------------------

def test_apply_rules_first_match_case_insensitive_and_missing_field():
    df = pd.DataFrame({"d": ["Basal cell carcinoma", "NEVUS, blue", None]})
    rules = [{"field": "d", "contains": "nevus", "diagnosis": "nevus"},
             {"field": "d", "contains": "carcinoma", "diagnosis": "bcc"},
             {"field": "missing_column", "contains": "x", "diagnosis": "never"}]
    assert harmonize.apply_rules(df, rules).tolist()[:2] == ["bcc", "nevus"]
    assert pd.isna(harmonize.apply_rules(df, rules).iloc[2])


def test_value_cleaners():
    sites = harmonize.map_body_site(pd.Series(["FACE", " Foot ", "elbow", None, "moon"]),
                                    {"head_neck": ["face"], "acral": ["foot"],
                                     "upper_extremity": ["elbow"]})
    assert sites.tolist() == ["head_neck", "acral", "upper_extremity", "unknown", "unknown"]
    assert harmonize.normalise_sex(pd.Series(["MALE", "Female", "unknown", None])).tolist() \
        == ["male", "female", "unknown", "unknown"]
    ages = harmonize.clean_age(pd.Series([40, -1, 130, "abc"]))
    assert ages.iloc[0] == 40 and ages.iloc[1:].isna().all()
    answers = harmonize.parse_answer(pd.Series(["True", "False", "UNK", None]))
    assert answers.iloc[:2].tolist() == [1.0, 0.0] and answers.iloc[2:].isna().all()


def test_attach_urgency_rejects_unmapped_diagnosis():
    df = pd.DataFrame({"diagnosis": ["melanoma", "brand_new"]})
    cfg = {"tiers": ["routine", "urgent"], "mapping": {"melanoma": "urgent"}}
    with pytest.raises(harmonize.HarmonizationError, match="brand_new"):
        harmonize.attach_urgency(df, cfg)


def test_every_configured_diagnosis_has_an_urgency(configs):
    harm_cfg, urgency_cfg = configs
    produced = set()
    for source_cfg in harm_cfg["diagnosis"].values():
        produced |= set(source_cfg.get("codes", {}).values())
        produced |= {rule["diagnosis"] for rule in source_cfg.get("rules", [])}
    assert produced <= set(urgency_cfg["mapping"])


def test_harmonize_all_maps_and_drops(tmp_path, configs):
    write_raw_datasets(tmp_path)
    df, report = harmonize.harmonize_all(tmp_path, *configs)
    assert list(df.columns) == harmonize.COMMON_COLUMNS
    assert report["isic"]["unmapped_diagnosis"] == 1          # "Unseen label"
    assert report["ham10000"]["missing_image"] == 1           # ISIC_0024308.jpg absent
    by_id = df.set_index("image_id")
    mel = by_id.loc["isic__ISIC_0000001"]
    assert (mel["diagnosis"], mel["urgency"], mel["body_site"]) == \
        ("melanoma", "urgent", "trunk")
    assert pd.isna(by_id.loc["isic__ISIC_0024306", "age"])   # 130 is impossible
    assert by_id.loc["isic__ISIC_0024306", "body_site"] == "acral"
    pad = by_id.loc["pad_ufes_20__PAT_1_10_1"]
    assert pad["q_itch"] == 1 and pd.isna(pad["q_changed"]) and pad["image_type"] == "smartphone"
    assert by_id.loc["fitzpatrick17k__aaa", "urgency"] == "urgent"   # other_malignant
    assert pd.isna(by_id.loc["fitzpatrick17k__bbb", "fitzpatrick"])  # -1 = unknown


def test_harmonize_all_missing_metadata(tmp_path, configs):
    with pytest.raises(harmonize.HarmonizationError, match="ingest"):
        harmonize.harmonize_all(tmp_path, *configs, datasets=["isic"])


# --- image_stats ------------------------------------------------------------------

def test_measure_image_detects_blur_darkness_and_corruption(tmp_path):
    sharp = image_stats.measure_image(str(textured_image(tmp_path / "s.jpg", seed=1)))
    blurry = image_stats.measure_image(str(textured_image(tmp_path / "b.jpg", seed=1, blur=4)))
    dark = image_stats.measure_image(str(textured_image(tmp_path / "d.jpg", seed=1,
                                                        brightness=0.1)))
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    broken = image_stats.measure_image(str(tmp_path / "bad.jpg"))
    missing = image_stats.measure_image(str(tmp_path / "nope.jpg"))
    assert sharp["readable"] and (sharp["width"], sharp["height"]) == (300, 260)
    assert blurry["sharpness"] < sharp["sharpness"] / 10
    assert dark["brightness"] < 25 < sharp["brightness"]
    assert len(sharp["phash"]) == 16
    assert not broken["readable"] and "UnidentifiedImageError" in broken["error"]
    assert not missing["readable"]


def test_laplacian_variance_flat_image_is_zero():
    assert image_stats.laplacian_variance(np.full((20, 20), 128, dtype=np.uint8)) == 0


# --- quality --------------------------------------------------------------------

QUALITY_CFG = {"min_side_px": 100, "min_brightness": 30, "max_brightness": 240,
               "min_contrast": 8, "min_sharpness": 10}


def test_quality_reasons_each_check_and_first_reason_wins():
    stats = pd.DataFrame({
        "readable": [True, False, True, True, True, True, True, True],
        "width": [300, np.nan, 50, 300, 300, 300, 300, 50],
        "height": [300, np.nan, 300, 300, 300, 300, 300, 300],
        "brightness": [120, np.nan, 120, 10, 250, 120, 120, 10],
        "contrast": [40, np.nan, 40, 40, 40, 2, 40, 40],
        "sharpness": [500, np.nan, 500, 500, 500, 500, 1, 500],
    })
    assert quality.quality_reasons(stats, QUALITY_CFG).tolist() == [
        "", "unreadable", "too_small", "too_dark", "too_bright", "low_contrast", "blurry",
        "too_small"]


def test_filter_quality_report():
    df = pd.DataFrame({"image_id": ["a", "b"], "source": ["isic", "isic"]})
    stats = pd.DataFrame({"image_id": ["a", "b"], "readable": [True, False],
                          "width": [300, np.nan], "height": [300, np.nan],
                          "brightness": [100, np.nan], "contrast": [30, np.nan],
                          "sharpness": [100, np.nan]})
    passed, dropped, report = quality.filter_quality(df, stats, QUALITY_CFG)
    assert passed["image_id"].tolist() == ["a"]
    assert report["isic"] == {"total": 2, "passed": 1, "dropped": {"unreadable": 1},
                              "dropped_share": 0.5}


# --- dedupe ----------------------------------------------------------------------

def test_near_duplicate_pairs_matches_brute_force():
    rnd = random.Random(0)
    base = [rnd.getrandbits(64) for _ in range(40)]
    values = base + [b ^ (1 << rnd.randrange(64)) ^ (1 << rnd.randrange(64)) for b in base[:15]]
    hashes = pd.Series([f"{v:016x}" for v in values])
    expected = sorted((i, j, (values[i] ^ values[j]).bit_count())
                      for i in range(len(values)) for j in range(i + 1, len(values))
                      if (values[i] ^ values[j]).bit_count() <= 4)
    assert dedupe.near_duplicate_pairs(hashes, 4) == expected


def _dedupe_frame():
    return pd.DataFrame({
        "image_id": ["isic__X1", "ham10000__X1", "ham10000__X2", "ham10000__X3",
                     "fitzpatrick17k__f", "isic__Y"],
        "source": ["isic", "ham10000", "ham10000", "ham10000", "fitzpatrick17k", "isic"],
        "source_image_id": ["X1", "X1", "X2", "X3", "f", "Y"],
        "patient_id": [np.nan] * 6,
        "lesion_id": [np.nan, "L1", "L2", "L3", np.nan, np.nan],
        "phash": ["ffffffffffffffff", "ffffffffffffffff", "0f0f0f0f0f0f0f0f",
                  "0f0f0f0f0f0f0f0e",   # 1 bit from X2: near-copy, same pool
                  "0f0f0f0f0f0f0f0c",   # 1 bit from X3 but other pool -> dropped
                  "123456789abcdef0"],
    })


def test_dedupe_drops_cross_dataset_copies_and_merges_groups():
    cfg = {"max_hamming_distance": 4,
           "source_priority": ["ham10000", "isic", "pad_ufes_20", "fitzpatrick17k"]}
    kept, dropped, _ = dedupe.dedupe(_dedupe_frame(), cfg)
    reasons = dict(zip(dropped["image_id"], dropped["drop_reason"]))
    assert reasons == {"isic__X1": "same_isic_id", "fitzpatrick17k__f": "near_copy_other_pool"}
    groups = kept.set_index("image_id")["group_id"]
    assert groups["ham10000__X2"] == groups["ham10000__X3"]          # merged near-copies
    assert groups["isic__Y"] != groups["ham10000__X2"]


def test_base_group_prefers_patient_then_lesion_then_image():
    df = pd.DataFrame({"source": ["s"] * 3, "patient_id": ["P", np.nan, np.nan],
                       "lesion_id": ["L", "L2", np.nan], "image_id": ["a", "b", "c"]})
    assert dedupe.base_group(df).tolist() == ["s/P", "s/L2", "s/c"]


# --- transform -------------------------------------------------------------------

def test_shades_of_gray_removes_colour_cast():
    rng = np.random.default_rng(0)
    neutral = rng.integers(60, 200, size=(50, 50, 1)).repeat(3, axis=2).astype(np.uint8)
    cast = np.clip(neutral * np.array([1.0, 0.8, 0.6]), 0, 255).astype(np.uint8)
    corrected = transform.shades_of_gray(cast).reshape(-1, 3).mean(axis=0)
    assert np.ptp(corrected) < np.ptp(cast.reshape(-1, 3).mean(axis=0)) / 5


def test_transform_image_output_size(tmp_path):
    src = textured_image(tmp_path / "in.png", size=(400, 250))
    assert transform.transform_image((str(src), str(tmp_path / "out.jpg")), 128, 6, 90) == ""
    assert Image.open(tmp_path / "out.jpg").size == (128, 128)
    assert "Error" in transform.transform_image((str(tmp_path / "x"), str(tmp_path / "y")),
                                                128, 6, 90)


def test_encode_features_marks_missing_instead_of_guessing():
    df = pd.DataFrame({"image_id": ["a", "b"], "age": [40, np.nan],
                       "sex": ["female", None], "body_site": ["trunk", "moon"],
                       **{c: [1.0, np.nan] for c in harmonize.ANSWER_COLUMNS}})
    out = transform.encode_features(df, [0, 30, 60, 120]).set_index("image_id")
    assert out.loc["b", "age_missing"] == 1 and out.loc["b", "age"] == 40
    assert out.loc["a", "age_bin_30_60"] == 1 and out.loc["b", "age_bin_30_60"] == 0
    assert out.loc["b", "sex_unknown"] == 1 and out.loc["b", "site_unknown"] == 0
    assert out.loc["a", "q_itch"] == 1 and out.loc["b", "q_itch_missing"] == 1
    assert out.loc["b", "q_itch"] == 0


# --- split ------------------------------------------------------------------------

def _split_frame(n_groups=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        tier = rng.choice(["urgent", "priority", "routine"], p=[0.15, 0.25, 0.6])
        for k in range(rng.integers(1, 4)):
            rows.append({"image_id": f"{g}_{k}", "group_id": f"g{g}", "urgency": tier,
                         "phash": f"{g:016x}", "source": "isic"})
    return pd.DataFrame(rows)


def test_assign_splits_ratios_groups_and_determinism():
    df = _split_frame()
    ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    df["split"] = split.assign_splits(df, ratios, seed=42)
    shares = df["split"].value_counts(normalize=True)
    assert abs(shares["train"] - 0.7) < 0.05 and abs(shares["test"] - 0.15) < 0.05
    assert (df.groupby("group_id")["split"].nunique() == 1).all()
    urgent = df[df["urgency"] == "urgent"]["split"].value_counts(normalize=True)
    assert abs(urgent["train"] - 0.7) < 0.1          # stratified by urgency
    assert df["split"].equals(split.assign_splits(df, ratios, seed=42))
    split.check_no_leakage(df)


def test_assign_splits_bad_ratios():
    with pytest.raises(ValueError):
        split.assign_splits(_split_frame(5), {"train": 0.8, "test": 0.3}, seed=1)


def test_check_no_leakage_detects_shared_group_and_hash():
    df = pd.DataFrame({"group_id": ["g1", "g1"], "phash": ["a", "b"],
                       "split": ["train", "test"]})
    with pytest.raises(split.LeakageError, match="group_id"):
        split.check_no_leakage(df)
    df = pd.DataFrame({"group_id": ["g1", "g2"], "phash": ["a", "a"],
                       "split": ["train", "test"]})
    with pytest.raises(split.LeakageError, match="phash"):
        split.check_no_leakage(df)


# --- whole preprocessing chain on the synthetic datasets -----------------------------

def test_preprocessing_end_to_end(params, tmp_path):
    write_raw_datasets(tmp_path / "raw")
    params["preprocess"]["workers"] = 1
    params["preprocess"]["transform"]["image_size"] = 64
    harmonize.run(params)
    image_stats.run(params)
    quality.run(params)
    summary = dedupe.run(params)
    assert summary["dropped"] == {"same_isic_id": 1}
    transform.run(params)
    split.run(params)

    dataset = pd.read_csv(tmp_path / "processed" / "dataset.csv")
    assert set(dataset["split"]) <= {"train", "val", "test", "pad_train", "pad_val",
                                     "pad_test", "fairness_test"}
    assert (dataset.loc[dataset["source"] == "fitzpatrick17k", "split"]
            == "fairness_test").all()
    pad = dataset[dataset["source"] == "pad_ufes_20"]
    assert pad["split"].nunique() == 1            # both PAD images belong to PAT_1
    for rel in dataset["processed_image"]:
        assert Image.open(tmp_path / "processed" / rel).size == (64, 64)
    assert (tmp_path / "logs" / "split_report.json").is_file()
