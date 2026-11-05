"""Check offline detector exports become safe, unannotated OCR crop templates."""

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from anpr.prepare import prepare_dataset


class DatasetPreparationTests(unittest.TestCase):
    """Verify crop geometry, label provenance, and fail-closed preparation."""

    def setUp(self):
        """Give each dataset an isolated source and destination directory."""
        # Temporary ownership keeps tests independent of downloaded user assets.
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "prepared"

    def tearDown(self):
        """Remove only the directory owned by this test."""
        # Crops and source images can be deleted together after assertions finish.
        self.workspace.cleanup()

    def make_source(self, split="train", name="frame.png", label="0 0.5 0.5 0.5 0.5"):
        """Write a small RGB source with a known foreground rectangle."""
        # A colored image makes grayscale conversion or annotation overlays visible.
        image_path = self.dataset / split / "images" / name
        label_path = self.dataset / split / "labels" / Path(name).with_suffix(".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        # Whole-image color is retained exactly by lossless crop PNGs.
        image = Image.new("RGB", (100, 40), (17, 89, 203))
        image.save(image_path)
        label_path.write_text(label, encoding="utf-8")
        return image_path

    def read_rows(self, manifest_path):
        """Read CSV rows with the same quoted-column behavior as training."""
        # Strings are retained to verify editable labels and portable relative paths.
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            return list(csv.DictReader(manifest_file))

    def test_crops_rgb_geometry_and_leaves_ocr_annotations_blank(self):
        """Detector box labels must never be mistaken for plate transcriptions."""
        self.make_source()
        # A centered half-size box maps to x=25..75 and y=10..30.
        manifest_path = prepare_dataset(self.dataset, self.output)
        rows = self.read_rows(manifest_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["plate"], "")
        self.assertEqual(rows[0]["group"], "")
        self.assertEqual(rows[0]["source"], "train/images/frame.png")
        self.assertEqual(rows[0]["box_index"], "0")
        self.assertEqual(
            [rows[0][column] for column in ("crop_x1", "crop_y1", "crop_x2", "crop_y2")],
            ["25", "10", "75", "30"],
        )
        with Image.open(self.output / rows[0]["path"]) as crop:
            self.assertEqual(crop.mode, "RGB")
            self.assertEqual(crop.size, (50, 20))
            self.assertEqual(crop.getpixel((0, 0)), (17, 89, 203))
        # Template labels must fail before any training can mistake them for supervision.
        from anpr.data import load_manifest

        with self.assertRaises(ValueError):
            load_manifest(manifest_path)

    def test_box_extents_are_clipped_to_source_image(self):
        """A normalized center at the edge may legally extend beyond the frame."""
        self.make_source(label="0 0 0 0.5 0.5")
        # The box extends into negative coordinates and must be clipped before slicing.
        rows = self.read_rows(prepare_dataset(self.dataset, self.output))
        with Image.open(self.output / rows[0]["path"]) as crop:
            self.assertEqual(crop.size, (25, 10))

    def test_same_names_in_different_splits_do_not_overwrite(self):
        """Source identity includes export split, even for identical filenames."""
        for split in ("train", "valid", "test"):
            # Repeated camera-frame filenames are common across exported partitions.
            self.make_source(split=split)
        rows = self.read_rows(prepare_dataset(self.dataset, self.output))
        self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})
        self.assertEqual(len({row["path"] for row in rows}), 3)
        self.assertTrue(all((self.output / row["path"]).is_file() for row in rows))

    def test_explicit_annotations_normalize_plate_and_keep_capture_group(self):
        """Only user-supplied OCR text and source groups fill the training labels."""
        self.make_source(label="1 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.5 0.5")
        # The line index refers to original detector annotation order, including other classes.
        annotations = self.root / "annotations.csv"
        annotations.write_text(
            "source,box_index,plate,group\ntrain/images/frame.png,1,ab12 cde,video-7\n", encoding="utf-8"
        )
        rows = self.read_rows(prepare_dataset(self.dataset, self.output, annotations=annotations))
        self.assertEqual(rows[0]["plate"], "AB12CDE")
        self.assertEqual(rows[0]["group"], "video-7")
        self.assertEqual(rows[0]["box_index"], "1")

    def test_partial_annotation_mapping_remains_an_explicit_template(self):
        """One transcription must not silently label other detected plates."""
        self.make_source(label="0 0.25 0.5 0.4 0.5\n0 0.75 0.5 0.4 0.5")
        # An unmatched crop remains blank so load_manifest rejects incomplete supervision.
        annotations = self.root / "annotations.csv"
        annotations.write_text(
            "source,box_index,plate,group\ntrain/images/frame.png,0,AB12CDE,camera-1\n", encoding="utf-8"
        )
        rows = self.read_rows(prepare_dataset(self.dataset, self.output, annotations=annotations))
        self.assertEqual([row["plate"] for row in rows], ["AB12CDE", ""])
        self.assertEqual([row["group"] for row in rows], ["camera-1", ""])

    def test_invalid_detector_labels_fail_before_writing_output(self):
        """Malformed coordinates and nonpositive extents must not create supervision."""
        # These independent cases cover range, finiteness, shape, and class validation.
        invalid_labels = [
            "0 nan 0.5 0.2 0.2",
            "0 0.5 0.5 0 0.2",
            "0 1.1 0.5 0.2 0.2",
            "0 0.5 0.5 -0.1 0.2",
            "0 0.5 0.5 0.2",
            "-1 0.5 0.5 0.2 0.2",
        ]
        for label in invalid_labels:
            # Reuse only the source fixture; a failed preparation cannot leave a destination.
            self.make_source(label=label)
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    prepare_dataset(self.dataset, self.output)
                self.assertFalse(self.output.exists())

    def test_existing_outputs_are_not_overwritten(self):
        """Re-running preparation must preserve crops and edited transcriptions."""
        self.make_source()
        # The original manifest is authoritative after a user starts labelling it.
        manifest_path = prepare_dataset(self.dataset, self.output)
        original_manifest = manifest_path.read_text(encoding="utf-8")
        with self.assertRaises(FileExistsError):
            prepare_dataset(self.dataset, self.output)
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), original_manifest)

    def test_unused_or_escaping_annotation_sources_are_rejected(self):
        """Typos and traversal paths in an explicit mapping must remain visible."""
        self.make_source()
        for source in ("train/images/missing.png", "../frame.png", "/frame.png"):
            # No valid source should receive an OCR label by a guessed basename match.
            annotations = self.root / "annotations.csv"
            annotations.write_text(
                f"source,box_index,plate,group\n{source},0,AB12CDE,camera-1\n", encoding="utf-8"
            )
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    prepare_dataset(self.dataset, self.output, annotations=annotations)
                self.assertFalse(self.output.exists())

    def test_symlinked_source_cannot_escape_dataset(self):
        """An extracted dataset cannot make crop preparation read unrelated paths."""
        self.make_source()
        # Replace a dataset image with a valid image outside its declared root.
        outside = self.root / "outside.png"
        Image.new("RGB", (10, 10), "white").save(outside)
        source = self.dataset / "train/images/frame.png"
        source.unlink()
        source.symlink_to(outside)
        with self.assertRaises(ValueError):
            prepare_dataset(self.dataset, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    # Running this file directly requires only Pillow and the preparation module.
    unittest.main()
