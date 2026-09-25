import torch

from oled_moe.memory import PinnedExpertStore


def test_expert_major_store_supports_single_expert_views() -> None:
    store = PinnedExpertStore(dtype=torch.float32, pin_memory=False)
    layer = store.register_layer(
        layer_id=2,
        num_experts=3,
        hidden_size=4,
        intermediate_size=2,
    )
    for expert_id in range(3):
        layer.put(expert_id, "gate_proj", torch.full((2, 4), expert_id + 1.0))
        layer.put(expert_id, "up_proj", torch.full((2, 4), expert_id + 2.0))
        layer.put(expert_id, "down_proj", torch.full((4, 2), expert_id + 3.0))
    store.validate_complete()
    assert layer.w13.shape == (3, 4, 4)
    assert layer.w2.shape == (3, 4, 2)
    assert layer.packed.shape == (3, 24)
    assert layer.packed.is_contiguous()
    assert torch.equal(layer.w13[1, :2], torch.full((2, 4), 2.0))
    assert torch.equal(layer.w2[1], torch.full((4, 2), 4.0))
    assert layer.w13[1].untyped_storage().data_ptr() == layer.w13.untyped_storage().data_ptr()
    assert layer.w13.untyped_storage().data_ptr() == layer.packed.untyped_storage().data_ptr()
    assert layer.w2.untyped_storage().data_ptr() == layer.packed.untyped_storage().data_ptr()
    assert layer.packed[1].is_contiguous()
