import torch

from dreamer.world_model import (
    WorldModel,
    WorldModelTrainingParams,
    EncoderParams,
    DecoderParams,
    RSSMParams,
    MLPandHeadParams,
)
from dreamer.networks.shared import BernouliDistParams, MLPParams, TwoHotDistParams
from dreamer.networks.encoder import CNNParams
from dreamer.networks.decoder import DecoderCNNParams

BATCH_SIZE = 4
BATCH_LENGTH = 8


def get_world_model() -> WorldModel:
    training_params = WorldModelTrainingParams()
    enc_cnn_params = CNNParams(
        initial_depth=24,
        kernel_size=3,
        stride=1,
        min_res=4,
        bias=True,
        norm=True,
        act_func="ReLU",
        max_pool=True,
        depth_mult=(2, 3, 4, 4),
        max_pool_stride=2,
        max_pool_kernel=1,
        dilation=1,
        image_shape=(64, 64, 3),
    )
    decoder_cnn_params = DecoderCNNParams(
        starting_res=4,
        latent_dim=264,
        kernel_size=3,
        starting_depth=24,
        depth_mults=(2, 3, 4, 4),
        bias=True,
        norm=True,
        act_func="ReLU",
        final_sigmoid=True,
        image_shape=(64, 64, 3),
    )
    enc_params = EncoderParams(
        image_keys=["image"], image_shapes=[(64, 64, 3)], cnn_params=enc_cnn_params
    )
    dec_params = DecoderParams(
        image_keys=["image"], image_shapes=[(64, 64, 3)], cnn_params=decoder_cnn_params
    )
    rssm_params = RSSMParams(
        deter_size=64,
        n_stoch_dists=10,
        n_stoch_cats=20,
        encoded_size=1536,
        hidden_size=64,
        act_func="ReLU",
        n_prior_layers=1,
        n_post_layers=1,
        n_deter_layers=1,
        layer_norm=True,
        bias=True,
        unimix=0.01,
        winit_scale=1.0,
        n_blocks=8,
        action_dim=10,
    )
    reward_network_params = MLPParams(
        input_dim=264,
        out_dim=255,
        n_layers=1,
        layer_width=64,
        layer_norm=True,
        bias=True,
        winit_scale=0.0,
        act_func="ReLU",
    )
    reward_head_params = TwoHotDistParams()
    continue_network_params = MLPParams(
        input_dim=264,
        out_dim=1,
        n_layers=1,
        layer_width=64,
        act_func="ReLU",
        layer_norm=True,
        bias=True,
        winit_scale=1.0,
    )
    continue_head_param = BernouliDistParams()

    grad_components = ["encoder", "decoder", "reward", "continue"]

    return WorldModel(
        training_params=training_params,
        rssm_params=rssm_params,
        encoder_params=enc_params,
        decoder_params=dec_params,
        grad_components=grad_components,
        reward_params=MLPandHeadParams(reward_network_params, reward_head_params),
        continue_params=MLPandHeadParams(continue_network_params, continue_head_param),
    )


def get_mock_data():
    images = torch.randn((BATCH_SIZE, BATCH_LENGTH, 64, 64, 3))
    reward = torch.randn((BATCH_SIZE, BATCH_LENGTH, 1))
    continues = torch.ones((BATCH_SIZE, BATCH_LENGTH, 1)).to(torch.float32)
    is_first = torch.zeros((BATCH_SIZE, BATCH_LENGTH, 1))
    is_first[:, 0] = 1
    is_first = is_first.to(torch.int32)
    actions = torch.zeros((BATCH_SIZE, BATCH_LENGTH, 10)).to(torch.float32)

    data = {
        "image": images,
        "reward": reward,
        "continue": continues,
        "is_first": is_first,
        "action": actions,
    }
    return data


def test_train():
    world_model = get_world_model()
    data = get_mock_data()

    loss_dict, _ = world_model.train(data)

    for loss_name, loss_val in loss_dict.items():
        assert loss_val.shape == ()
        assert loss_val <= 0.0


def test_zero_reward():
    world_model = get_world_model()
    mock_latent = torch.randn((BATCH_SIZE, BATCH_LENGTH, 264))
    assert world_model._reward_network is not None
    assert world_model._reward_head is not None
    reward_logits = world_model._reward_network(mock_latent)
    reward_dist = world_model._reward_head(reward_logits)

    reward_pred = reward_dist.predict()
    assert torch.all(reward_pred == 0.0)
