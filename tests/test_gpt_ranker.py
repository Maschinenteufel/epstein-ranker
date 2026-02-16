import argparse
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gpt_ranker


class GptRankerHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.skip_args = argparse.Namespace(
            min_text_chars=60,
            min_text_words=12,
            min_alpha_ratio=0.25,
            min_unique_word_ratio=0.15,
            max_short_token_ratio=0.6,
            min_avg_word_length=3.0,
            min_long_word_count=4,
            max_repeated_char_run=40,
            include_action_items=False,
            justice_files_base_url=gpt_ranker.DEFAULT_JUSTICE_FILES_BASE_URL,
        )

    def test_iter_rows_supports_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            (nested / "first.txt").write_text("hello world", encoding="utf-8")
            (root / "second.txt").write_text("another file", encoding="utf-8")
            (root / "ignore.md").write_text("should not load", encoding="utf-8")

            rows = list(gpt_ranker.iter_rows(root, input_glob="*.txt", include_text=True))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["filename"], "a/b/first.txt")
        self.assertEqual(rows[1]["filename"], "second.txt")
        self.assertEqual(rows[0]["text"], "hello world")

    def test_iter_rows_supports_csv_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "data.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename", "text"])
                writer.writeheader()
                writer.writerow({"filename": "one.txt", "text": "hello"})
                writer.writerow({"filename": "two.txt", "text": "world"})

            rows = list(gpt_ranker.iter_rows(csv_path))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["filename"], "one.txt")
        self.assertEqual(rows[1]["text"], "world")

    def test_iter_rows_supports_image_mode_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_pdf = root / "doc.pdf"
            image_pdf.write_bytes(b"%PDF-1.4\n")
            rows = list(
                gpt_ranker.iter_rows(
                    root,
                    input_glob="*.pdf",
                    include_text=True,
                    processing_mode="image",
                )
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_kind"], "image")
        self.assertEqual(rows[0]["text"], "")

    def test_iter_rows_splits_pdf_into_parts_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_pdf = root / "doc.pdf"
            image_pdf.write_bytes(b"%PDF-1.4\n")
            with mock.patch.object(gpt_ranker, "detect_pdf_page_count", return_value=10):
                rows = list(
                    gpt_ranker.iter_rows(
                        root,
                        input_glob="*.pdf",
                        include_text=False,
                        processing_mode="image",
                        pdf_part_pages=4,
                    )
                )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["part_index"], 1)
        self.assertEqual(rows[0]["part_total"], 3)
        self.assertEqual(rows[0]["part_start_page"], 1)
        self.assertEqual(rows[0]["part_end_page"], 4)
        self.assertIn("::part_0001_p00001-00004", rows[0]["source_id"])
        self.assertEqual(rows[1]["part_start_page"], 5)
        self.assertEqual(rows[1]["part_end_page"], 8)
        self.assertEqual(rows[2]["part_start_page"], 9)
        self.assertEqual(rows[2]["part_end_page"], 10)

    def test_skip_reason_flags_low_quality_rows(self) -> None:
        quality = gpt_ranker.assess_text_quality("x")
        reason = gpt_ranker.build_skip_reason(quality, self.skip_args)
        self.assertIsNotNone(reason)
        self.assertIn("too short", reason)

    def test_skip_reason_flags_ocr_noise_tokens(self) -> None:
        text = "~ © B S)Geeeee ee A go 6 4 ls * . ; errr : id ° N + oO oO oO oO oO oO << = Ww Lu."
        quality = gpt_ranker.assess_text_quality(text)
        reason = gpt_ranker.build_skip_reason(quality, self.skip_args)
        self.assertIsNotNone(reason)
        self.assertTrue(
            "too many short/noisy tokens" in reason
            or "average token length too low/noisy OCR" in reason
        )

    def test_skip_reason_allows_normal_text(self) -> None:
        text = (
            "This is a normal paragraph with enough words to pass screening and "
            "contains meaningful language for downstream analysis."
        )
        quality = gpt_ranker.assess_text_quality(text)
        reason = gpt_ranker.build_skip_reason(quality, self.skip_args)
        self.assertIsNone(reason)

    def test_build_output_records_marks_skipped_status(self) -> None:
        source_row = {"filename": "sample.txt", "text": ""}
        quality = gpt_ranker.assess_text_quality("")
        result = gpt_ranker.build_skipped_model_result("empty text")
        csv_row, json_record = gpt_ranker.build_output_records(
            idx=3,
            source_row=source_row,
            result=result,
            args=self.skip_args,
            config_metadata={"model": "qwen/qwen3-coder-next"},
            quality=quality,
            processing_status="skipped",
            skip_reason="empty text",
        )

        self.assertEqual(csv_row["processing_status"], "skipped")
        self.assertEqual(csv_row["skip_reason"], "empty text")
        self.assertEqual(csv_row["importance_score"], 0)
        self.assertEqual(json_record["metadata"]["processing_status"], "skipped")
        self.assertEqual(json_record["metadata"]["skip_reason"], "empty text")
        self.assertEqual(csv_row["source_pdf_url"], "")

    def test_build_output_records_includes_part_metadata(self) -> None:
        source_row = {
            "filename": "VOL00003/IMAGES/0001/EFTA00000001.pdf",
            "source_id": "VOL00003/IMAGES/0001/EFTA00000001.pdf::part_0002_p00025-00048",
            "document_part": "part_0002_of_0005_p00025-00048",
            "part_index": 2,
            "part_total": 5,
            "part_start_page": 25,
            "part_end_page": 48,
            "document_total_pages": 117,
            "text": "",
        }
        result = {
            "headline": "h",
            "importance_score": 42,
            "reason": "r",
            "key_insights": [],
            "tags": [],
            "power_mentions": [],
            "agency_involvement": [],
            "lead_types": [],
        }
        csv_row, json_record = gpt_ranker.build_output_records(
            idx=10,
            source_row=source_row,
            result=result,
            args=self.skip_args,
            config_metadata={"model": "qwen/qwen3-vl-30b"},
            quality={},
            processing_status="processed",
            skip_reason="",
        )

        self.assertEqual(
            csv_row["source_id"],
            "VOL00003/IMAGES/0001/EFTA00000001.pdf::part_0002_p00025-00048",
        )
        self.assertEqual(csv_row["document_part"], "part_0002_of_0005_p00025-00048")
        self.assertEqual(csv_row["part_index"], 2)
        self.assertEqual(csv_row["part_total"], 5)
        self.assertEqual(json_record["document_total_pages"], 117)
        self.assertEqual(json_record["metadata"]["part_start_page"], 25)

    def test_derive_justice_pdf_url_from_dataset_path(self) -> None:
        filename = "DataSet10/IMAGES/0332/EFTA01970985.txt"
        url = gpt_ranker.derive_justice_pdf_url(filename)
        self.assertEqual(
            url,
            "https://www.justice.gov/epstein/files/DataSet%2010/EFTA01970985.pdf",
        )

    def test_derive_justice_pdf_url_returns_none_when_unmatched(self) -> None:
        self.assertIsNone(gpt_ranker.derive_justice_pdf_url("notes/no_match.txt"))

    def test_derive_justice_pdf_url_from_volume_path(self) -> None:
        url = gpt_ranker.derive_justice_pdf_url(
            "IMAGES/0001/EFTA00000001.pdf",
            source_path="/tmp/data/new_data/VOL00001/IMAGES/0001/EFTA00000001.pdf",
        )
        self.assertEqual(
            url,
            "https://www.justice.gov/epstein/files/DataSet%201/EFTA00000001.pdf",
        )

    def test_derive_justice_pdf_url_from_dataset_tag(self) -> None:
        url = gpt_ranker.derive_justice_pdf_url(
            "IMAGES/0001/EFTA00000001.pdf",
            dataset_tag="standardworks_epstein_files_vol00001",
        )
        self.assertEqual(
            url,
            "https://www.justice.gov/epstein/files/DataSet%201/EFTA00000001.pdf",
        )

    def test_derive_local_source_url_maps_data_path(self) -> None:
        source_url = gpt_ranker.derive_local_source_url(
            "/tmp/project/source/data/new_data/VOL00001/IMAGES/0001/EFTA00000001.pdf",
            "IMAGES/0001/EFTA00000001.pdf",
            source_files_base_url=None,
        )
        self.assertEqual(
            source_url,
            "/data/new_data/VOL00001/IMAGES/0001/EFTA00000001.pdf",
        )

    def test_cli_explicit_default_value_overrides_config(self) -> None:
        original_argv = sys.argv[:]
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                config_path = Path(tmpdir) / "ranker_config.toml"
                config_path.write_text("sleep = 0.5\n", encoding="utf-8")
                sys.argv = [
                    "gpt_ranker.py",
                    "--config",
                    str(config_path),
                    "--sleep",
                    "0",
                ]
                args = gpt_ranker.parse_args()
                self.assertEqual(args.sleep, 0.0)
        finally:
            sys.argv = original_argv

    def test_apply_dataset_workspace_defaults_sets_isolated_paths(self) -> None:
        args = argparse.Namespace(
            dataset_workspace_root=Path("data/workspaces"),
            dataset_tag=None,
            input=Path("data/new_data"),
            output=Path("data/epstein_ranked.csv"),
            json_output=Path("data/epstein_ranked.jsonl"),
            checkpoint=Path("data/.epstein_checkpoint"),
            chunk_dir=Path("contrib"),
            chunk_manifest=Path("data/chunks.json"),
            run_metadata_file=None,
            known_json=["old.jsonl"],
        )
        gpt_ranker.apply_dataset_workspace_defaults(args, cli_explicit=set())

        self.assertEqual(args.dataset_tag, "new_data")
        self.assertEqual(
            args.output,
            Path("data/workspaces/new_data/results/epstein_ranked.csv"),
        )
        self.assertEqual(
            args.chunk_manifest,
            Path("data/workspaces/new_data/metadata/chunks.json"),
        )
        self.assertEqual(args.known_json, [])

    def test_apply_dataset_workspace_defaults_keeps_explicit_paths(self) -> None:
        args = argparse.Namespace(
            dataset_workspace_root=Path("data/workspaces"),
            dataset_tag="custom",
            input=Path("data/new_data"),
            output=Path("/tmp/custom.csv"),
            json_output=Path("/tmp/custom.jsonl"),
            checkpoint=Path("/tmp/custom.ckpt"),
            chunk_dir=Path("/tmp/chunks"),
            chunk_manifest=Path("/tmp/chunks.json"),
            run_metadata_file=Path("/tmp/run_meta.json"),
            known_json=["keep.jsonl"],
        )
        gpt_ranker.apply_dataset_workspace_defaults(
            args,
            cli_explicit={
                "output",
                "json_output",
                "checkpoint",
                "chunk_dir",
                "chunk_manifest",
                "run_metadata_file",
                "known_json",
            },
        )

        self.assertEqual(args.dataset_tag, "custom")
        self.assertEqual(args.output, Path("/tmp/custom.csv"))
        self.assertEqual(args.chunk_dir, Path("/tmp/chunks"))
        self.assertEqual(args.known_json, ["keep.jsonl"])

    def test_load_resume_completed_ids_ignores_checkpoint_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = root / ".checkpoint"
            checkpoint.write_text("IMAGES/0001/EFTA00000001.pdf\n", encoding="utf-8")
            args = argparse.Namespace(
                resume=True,
                checkpoint=checkpoint,
                json_output=root / "results.jsonl",
                known_json=[],
                chunk_size=1000,
                chunk_dir=root / "chunks",
            )
            completed = gpt_ranker.load_resume_completed_ids(args)
        self.assertEqual(completed, set())

    def test_load_resume_completed_ids_uses_chunk_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunk_dir = root / "chunks"
            chunk_dir.mkdir(parents=True)
            chunk_file = chunk_dir / "epstein_ranked_00001_01000.jsonl"
            chunk_file.write_text(
                '{"source_id":"IMAGES/0001/EFTA00000001.pdf","filename":"IMAGES/0001/EFTA00000001.pdf"}\n',
                encoding="utf-8",
            )
            checkpoint = root / ".checkpoint"
            checkpoint.write_text("STALE_ROW_ID\n", encoding="utf-8")
            args = argparse.Namespace(
                resume=True,
                checkpoint=checkpoint,
                json_output=root / "results.jsonl",
                known_json=[],
                chunk_size=1000,
                chunk_dir=chunk_dir,
            )
            completed = gpt_ranker.load_resume_completed_ids(args)
        self.assertEqual(completed, {"IMAGES/0001/EFTA00000001.pdf"})

    def test_write_run_metadata_writes_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_file = Path(tmpdir) / "meta" / "run.json"
            args = argparse.Namespace(
                run_metadata_file=run_file,
                dataset_tag="standardworks_epstein_files",
                dataset_source_label="StandardWorks",
                dataset_source_url="https://standardworks.ai/epstein-files",
                input=Path("data/new_data"),
                input_glob="*.txt",
                output=Path("out.csv"),
                json_output=Path("out.jsonl"),
                checkpoint=Path("ckpt"),
                chunk_size=1000,
                chunk_dir=Path("chunks"),
                chunk_manifest=Path("chunks.json"),
            )
            gpt_ranker.write_run_metadata(
                args=args,
                prompt_source="prompts/default_system_prompt.txt",
                config_metadata={"model": "qwen/qwen3-coder-next"},
                workload_stats={"total": 10, "already_done": 0, "workload": 10},
                total_dataset_rows=10,
                dataset_profile={"profile_id": "standardworks_epstein_files"},
            )

            payload = run_file.read_text(encoding="utf-8")
            self.assertIn("standardworks_epstein_files", payload)
            self.assertIn("dataset_profile", payload)

    def test_build_request_targets_adds_localhost_fallback(self) -> None:
        targets = gpt_ranker.build_request_targets(
            "http://localhost:5002/v1",
            "auto",
        )
        self.assertIn(("openai", "http://localhost:5002/v1"), targets)
        self.assertIn(("chat", "http://localhost:5002/v1"), targets)
        self.assertIn(("chat", "http://localhost:5555/api/v1"), targets)

    def test_call_model_chat_mode_parses_output(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        with mock.patch.object(gpt_ranker, "post_request", return_value=response_payload) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertTrue(mocked_post.call_args.kwargs["url"].endswith("/chat"))

    def test_call_model_auto_falls_back_to_chat_endpoint(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }

        def side_effect(*, url, payload, api_key, extra_headers, timeout):
            if url.endswith("/chat/completions"):
                raise gpt_ranker.UnsupportedEndpointError("no completions route")
            return response_payload

        with mock.patch.object(gpt_ranker, "post_request", side_effect=side_effect) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:1234/v1",
                api_format="auto",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["importance_score"], 1)
        called_urls = [call.kwargs["url"] for call in mocked_post.call_args_list]
        self.assertIn("http://localhost:1234/v1/chat/completions", called_urls)
        self.assertIn("http://localhost:1234/v1/chat", called_urls)

    def test_call_model_retries_transient_errors(self) -> None:
        response_payload = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        side_effects = [
            gpt_ranker.ModelRequestError("temporary outage", retriable=True),
            response_payload,
        ]
        with mock.patch.object(gpt_ranker, "post_request", side_effect=side_effects) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=2,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertEqual(mocked_post.call_count, 2)

    def test_call_model_retries_malformed_json_output(self) -> None:
        malformed = {
            "output": [
                {
                    "type": "message",
                    "content": '{"headline":"h","importance_score":1,"reason":"r"',
                }
            ]
        }
        valid = {
            "output": [
                {
                    "type": "message",
                    "content": (
                        '{"headline":"h","importance_score":1,"reason":"r",'
                        '"key_insights":[],"tags":[],"power_mentions":[],'
                        '"agency_involvement":[],"lead_types":[]}'
                    ),
                }
            ]
        }
        with mock.patch.object(gpt_ranker, "post_request", side_effect=[malformed, valid]) as mocked_post:
            result = gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-coder-next",
                filename="DataSet10/EFTA00000001.txt",
                text="Some useful text with enough detail for scoring.",
                input_kind="text",
                image_path=None,
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=2,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )
        self.assertEqual(result["headline"], "h")
        self.assertEqual(mocked_post.call_count, 2)

    def test_call_model_image_mode_rejects_chat_api(self) -> None:
        with self.assertRaises(RuntimeError):
            gpt_ranker.call_model(
                endpoint="http://localhost:5555/api/v1",
                api_format="chat",
                model="qwen/qwen3-vl-30b",
                filename="IMAGES/0001/EFTA00000001.pdf",
                text="",
                input_kind="image",
                image_path=Path("/tmp/nope.pdf"),
                image_max_pages=1,
                image_render_dpi=180,
                system_prompt="Return JSON",
                api_key=None,
                timeout=30,
                max_retries=1,
                retry_backoff=0,
                temperature=0.0,
                max_output_tokens=900,
                reasoning_effort=None,
                image_detail="low",
                config_metadata=None,
            )


if __name__ == "__main__":
    unittest.main()
