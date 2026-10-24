import unittest

from sglang.srt.parser.reasoning_parser import Glm45Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlm45ToolCode(unittest.TestCase):
    def test_code_literals_after_reasoning_every_partition(self):
        code = '<tool_call>edit<arg_key>text</arg_key><arg_value>print("<think></think>")</arg_value></tool_call>'
        for opener in ("", "<think>"):
            for closer in ("", "</think>"):
                text = opener + "plan" + closer + code
                for stream_reasoning in (False, True):
                    with self.subTest(
                        opener=opener, closer=closer, stream_reasoning=stream_reasoning
                    ):
                        complete = Glm45Detector(force_reasoning=True).detect_and_parse(
                            text
                        )
                        self.assertEqual(
                            (complete.reasoning_text, complete.normal_text),
                            ("plan", code),
                        )
                        partitions = [
                            [text[:i], text[i:]] for i in range(len(text) + 1)
                        ] + [list(text)]
                        for chunks in partitions:
                            parser = Glm45Detector(
                                force_reasoning=True, stream_reasoning=stream_reasoning
                            )
                            results = [
                                parser.parse_streaming_increment(chunk)
                                for chunk in chunks
                            ]
                            results.append(parser.finish())
                            self.assertEqual(
                                "".join(r.reasoning_text for r in results), "plan"
                            )
                            self.assertEqual(
                                "".join(r.normal_text for r in results), code
                            )

    def test_no_reasoning_tool_code_is_literal(self):
        text = '<tool_call>edit<arg_key>text</arg_key><arg_value>"<think>"</arg_value></tool_call>'
        parser = Glm45Detector(force_reasoning=False)
        result = parser.detect_and_parse(text)
        self.assertEqual((result.reasoning_text, result.normal_text), ("", text))
        result = parser.parse_streaming_increment(text)
        self.assertEqual((result.reasoning_text, result.normal_text), ("", text))


if __name__ == "__main__":
    unittest.main()
