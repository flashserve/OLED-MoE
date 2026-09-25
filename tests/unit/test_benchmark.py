import torch

from oled_moe.cli.benchmark import _generated_token_count


def test_generated_token_count_excludes_prompt_alignment_padding() -> None:
    input_ids = torch.zeros((1, 55), dtype=torch.long)
    output = torch.zeros((1, 192), dtype=torch.long)

    assert (
        _generated_token_count(
            output,
            input_ids,
            block_length=16,
            gen_length=128,
        )
        == 128
    )
