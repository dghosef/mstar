import sys, torch
sys.path.insert(0, "/root/guest-qwen35vl/rustext"); sys.path.insert(1, "/root/guest-qwen35vl/main")
torch.set_num_threads(4)
from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel
from transformers.video_utils import VideoMetadata
m = Qwen3_5DenseModel("Qwen/Qwen3.5-0.8B")
vp = m.video_processor
print(type(vp).__name__, vp.rescale_factor, vp.image_mean, vp.image_std, vp.temporal_patch_size, vp.patch_size)
ip = m.image_processor
print("image", ip.image_mean, ip.image_std, ip.temporal_patch_size)
out = m.load_video("/root/guest-qwen35vl/vids/t8s_360p.mp4", "cpu")
frames, md = out.data, out.metadata
ref = vp(videos=[frames.cpu()], video_metadata=[VideoMetadata(**md)], do_sample_frames=False, return_tensors="pt")
u8 = vp(videos=[frames.cpu()], video_metadata=[VideoMetadata(**md)], do_sample_frames=False, return_tensors="pt", do_rescale=False, do_normalize=False)
pv = u8["pixel_values_videos"]; print(pv.dtype, pv.shape, ref["pixel_values_videos"].dtype)
scale = 1.0 / vp.rescale_factor
mean = (torch.tensor(vp.image_mean) * scale).repeat_interleave(vp.temporal_patch_size * vp.patch_size ** 2)
std = (torch.tensor(vp.image_std) * scale).repeat_interleave(vp.temporal_patch_size * vp.patch_size ** 2)
print("equal", torch.equal((pv.to(torch.float32) - mean) / std, ref["pixel_values_videos"]), torch.equal(u8["video_grid_thw"], ref["video_grid_thw"]))
