import csv
import io
import unittest

from text_sources import (
    MAX_CSV_BYTES,
    find_text_column,
    read_texts_from_csv,
    render_batch_results_csv,
)


class TextSourcesTest(unittest.TestCase):
    def test_finds_tweet_text_column(self) -> None:
        batch = read_texts_from_csv('Tweet Text,Status\n"Great update",ok\n"",skip\n')

        self.assertEqual(batch.column, "Tweet Text")
        self.assertEqual(batch.texts, ("Great update",))
        self.assertFalse(batch.truncated)

    def test_accepts_normalized_aliases(self) -> None:
        batch = read_texts_from_csv(
            "tweet_text,score\nUseful   feature,1\nSecond row,0\n",
            limit=1,
        )

        self.assertEqual(batch.column, "tweet_text")
        self.assertEqual(batch.texts, ("Useful feature",))
        self.assertTrue(batch.truncated)

    def test_reports_missing_text_column(self) -> None:
        with self.assertRaisesRegex(ValueError, "text column"):
            read_texts_from_csv("rating,value\n5,10\n")

    def test_finds_existing_alias(self) -> None:
        self.assertEqual(find_text_column(["id", "Comments", "score"]), "Comments")

    def test_uses_documented_alias_priority(self) -> None:
        self.assertEqual(
            find_text_column(["comment", "review", "Tweet Text"]),
            "Tweet Text",
        )

    def test_rejects_oversized_csv_before_parsing(self) -> None:
        with self.assertRaisesRegex(ValueError, "5 MiB"):
            read_texts_from_csv(b"x" * (MAX_CSV_BYTES + 1))

    def test_download_neutralizes_spreadsheet_formulas(self) -> None:
        rendered = render_batch_results_csv(
            [
                {
                    "review": '=HYPERLINK("https://example.com")',
                    "sentiment": "Positive",
                    "confidence": "99%",
                }
            ]
        )
        rows = list(csv.DictReader(io.StringIO(rendered)))

        self.assertEqual(rows[0]["review"], '\'=HYPERLINK("https://example.com")')


if __name__ == "__main__":
    unittest.main()
