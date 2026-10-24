"""GLM XML round trips and stream partitions relevant to coding clients."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def xml_call(name, arguments):
    return (
        "<tool_call>"
        + name
        + "".join(
            f"<arg_key>{key}</arg_key><arg_value>"
            + (
                value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False)
            )
            + "</arg_value>"
            for key, value in arguments.items()
        )
        + "</tool_call>"
    )


class TestGlm47OpenCode(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                function=Function(
                    name="edit",
                    parameters={
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "data": {"type": "object"},
                            "items": {"type": "array"},
                            "ok": {"type": "boolean"},
                        },
                    },
                )
            ),
            Tool(
                function=Function(
                    name="pwd", parameters={"type": "object", "properties": {}}
                )
            ),
        ]

    def collect(self, chunks):
        detector = Glm47MoeDetector()
        calls, text, names = {}, "", {}
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            text += result.normal_text
            for item in result.calls:
                if item.name is not None:
                    self.assertNotIn(
                        item.tool_index, names, "name/ID must only be emitted once"
                    )
                    names[item.tool_index] = item.name
                self.assertIn(
                    item.tool_index, names, "arguments must follow their name"
                )
                calls[item.tool_index] = (
                    calls.get(item.tool_index, "") + item.parameters
                )
        end = detector.finish(self.tools)
        text += end.normal_text
        self.assertFalse(end.calls)
        self.assertEqual(detector.finish(self.tools).normal_text, "")
        return text, [(i, names[i], json.loads(args)) for i, args in calls.items()]

    def test_parallel_calls_every_two_way_partition(self):
        text = (
            "before "
            + xml_call("edit", {"text": "first"})
            + " between "
            + xml_call("pwd", {})
            + xml_call("edit", {"text": "second"})
            + " after"
        )
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                self.assertEqual(
                    self.collect([text[:split], text[split:]]),
                    (
                        "before  between  after",
                        [
                            (0, "edit", {"text": "first"}),
                            (1, "pwd", {}),
                            (2, "edit", {"text": "second"}),
                        ],
                    ),
                )

    def test_many_calls_in_one_chunk(self):
        # Iterative draining must not depend on Python's recursion limit.
        _, calls = self.collect([xml_call("pwd", {}) * 1100])
        self.assertEqual([i for i, _, _ in calls], list(range(1100)))

    def test_raw_string_round_trip(self):
        # The GLM chat template writes strings verbatim, without JSON quoting.
        for value in (
            '"quoted"',
            "'quoted'",
            r'"\n"',
            r'"\C|\."',
            "  def f():\n\treturn '中文'  \n",
            "",
            "null",
            "123_456",
            "<",
            "x <",
            "</arg_",
            "a<<",
        ):
            text = xml_call("edit", {"text": value})
            with self.subTest(value=value):
                complete = Glm47MoeDetector().detect_and_parse(text, self.tools).calls
                self.assertEqual(json.loads(complete[0].parameters), {"text": value})
                for size in (1, 7, len(text)):
                    self.assertEqual(
                        self.collect(
                            [text[i : i + size] for i in range(0, len(text), size)]
                        )[1],
                        [(0, "edit", {"text": value})],
                    )

    def test_structured_recovery_matches_nonstream(self):
        for key, raw, expected in (
            ("data", r"{\"a\": 1}", {"a": 1}),
            ("items", r"[\"a\", \"b\"]", ["a", "b"]),
            ("ok", "True", True),
            ("data", '{"nested":{"x":1}}', {"nested": {"x": 1}}),
        ):
            text = xml_call("edit", {key: raw})
            with self.subTest(key=key, raw=raw):
                complete = Glm47MoeDetector().detect_and_parse(text, self.tools).calls
                self.assertEqual(json.loads(complete[0].parameters), {key: expected})
                for size in (1, 11, len(text)):
                    self.assertEqual(
                        self.collect(
                            [text[i : i + size] for i in range(0, len(text), size)]
                        )[1],
                        [(0, "edit", {key: expected})],
                    )

    def test_function_whitespace_nonstream(self):
        result = Glm47MoeDetector().detect_and_parse(
            xml_call("\n edit\n", {"text": "x"}), self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "edit")

    def test_nonstream_indices_follow_calls_not_tool_definitions(self):
        text = (
            " before "
            + xml_call("pwd", {})
            + xml_call("pwd", {})
            + xml_call("edit", {"text": "x"})
            + " after "
        )
        result = Glm47MoeDetector().detect_and_parse(text, self.tools)
        self.assertEqual([call.tool_index for call in result.calls], [0, 1, 2])
        self.assertEqual(result.normal_text, " before  after ")

    def test_finish_releases_plain_partial_marker(self):
        for text in ("hello <", "a <tool_", "a </tool_call>"):
            with self.subTest(text=text):
                self.assertEqual(self.collect([text]), (text, []))

    def test_finish_does_not_complete_truncated_tool(self):
        detector = Glm47MoeDetector()
        result = detector.parse_streaming_increment(
            "<tool_call>edit<arg_key>text</arg_key><arg_value>abc", self.tools
        )
        before = "".join(item.parameters for item in result.calls)
        with self.assertRaises(json.JSONDecodeError):
            json.loads(before)
        end = detector.finish(self.tools)
        self.assertEqual((end.normal_text, end.calls), ("", []))


if __name__ == "__main__":
    unittest.main()
