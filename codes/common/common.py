import torch
from torch.utils.data import DataLoader, Dataset

import resource

def get_free_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory  # KiB


def get_rlimits():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    return soft, hard


def set_memory_limit(factor = 1):
    mem = get_free_memory()*1024
    _, hard = get_rlimits()
    resource.setrlimit(resource.RLIMIT_AS, (int(mem*factor), hard))

def set_memory_limit_if_not_limit(factor = 1):
    mem = get_free_memory()*1024
    soft, hard = get_rlimits()
    if soft == -1:
        resource.setrlimit(resource.RLIMIT_AS, (int(mem*factor), hard))

class VideoDataset(Dataset):

    def __init__(self, videos, target_videos=None, transform=None, input_target_cut = -1, stack_videos: bool = False, device='cpu'):
        self.videos = videos
        self.target_videos = target_videos
        self.input_target_cut = input_target_cut
        self.transform = transform
        self.stack_videos = stack_videos
        self.device = device

    def __len__(self):
        return len(self.videos)

    def ___getitem__(self, idx):
        video = self.videos[idx].values
        if self.transform is not None:
            video = self.apply_transform(video)

        if self.target_videos is not None:
            targ_vid = self.target_videos[idx].values
            if self.transform is not None:
                targ_vid = self.apply_transform(targ_vid)
            return video, targ_vid
        
        # Assume each video has N frames, input and target will be N-1 and 1 respectively
        input_seq = video[:self.input_target_cut]
        target_seq = video[self.input_target_cut:]
        return input_seq, target_seq

    def __getitem__(self, idx):
        vid, targ = self.___getitem__(idx)
        if self.stack_videos:
            return torch.cat([vid, targ]).to(self.device)
        return vid.to(self.device), targ.to(self.device)
    
    def apply_transform(self, video):
        seed = torch.seed()
        transformed_video = []
        for frame in video:
            # Set the same seed for each frame to ensure consistent transformations
            torch.manual_seed(seed)
            transformed_frame = self.transform(frame)
            transformed_video.append(transformed_frame)
        return torch.stack(transformed_video)
