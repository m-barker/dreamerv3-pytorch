from typing import List, Dict, Optional
import os
import random

import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyMemmapStorage, LazyTensorStorage


class Buffer:
    def __init__(
        self,
        capacity: int,
        keys_to_sample: List[str],
        disk_path: str,
        load_existing: bool = False,
        save_every: int = 10000,
    ) -> None:
        """
        Args:
            capacity (int): maximum number of transitions. Replaces old in a FIFO manner.

            keys_to_sample (List[str]): names of the keys that should be sampled from the buffer.

            disk_path (str): the path to where the replay buffer should be stored on disk.

            load_existing (bool, optional): whether to load an existing replay buffer. If True,
            attempts to load the "disk_path" into the buffer. Defaults to False.

            save_every (int, optional): save the buffer to the disk after x new transitions.
            Defaults to 10,000
        """

        self._buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=capacity))
        self._keys = keys_to_sample

        # We create this key to transitions to assign a UUID to each episode
        if "episode_id" not in self._keys:
            self._keys.append("episode_id")

        if load_existing:
            self.load_compact(disk_path)
            # self._buffer.loads(disk_path)
            print(f"Replay buffer loaded with {len(self._buffer)} steps")

        os.makedirs(disk_path, exist_ok=True)

        self._disk_path = disk_path
        self._save_every = save_every
        # Used to assign a UUID to each episode
        self._episode_id = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def _sample(self, batch_size: int, batch_length: int):
        sequences = []
        total = len(self._buffer)

        # Sample each sequence
        for _ in range(batch_size):
            chunks = []
            remaining = batch_length
            first_chunk = True

            while remaining > 0:
                if first_chunk:
                    # -2 to ensure at least 2 datapoints if hitting end of buffer
                    start_index = random.randint(0, total - 2)
                    first_chunk = False
                else:
                    # Move the start index to the start of the next episode
                    start_index = start_index + len(chunks[-1]["episode_id"])
                    # Edge case for exceeding buffer size, circle back to start
                    if start_index > total - 2:
                        start_index = 0

                start_episode_idx = self._buffer.storage["episode_id"][start_index]
                end_index = min(start_index + remaining, total - 1)

                while int(self._buffer.storage["episode_id"][end_index - 1]) != int(
                    start_episode_idx
                ):
                    end_index -= 1

                take_len = end_index - start_index

                td_slice = TensorDict(
                    {
                        k: self._buffer.storage[k][
                            start_index : start_index + take_len
                        ].clone()
                        for k in self._keys
                    },
                    batch_size=[take_len],
                )
                chunks.append(TensorDict(td_slice, batch_size=[take_len]))

                remaining -= take_len
            single_batch = torch.cat(chunks, dim=0)
            single_batch["is_first"][0] = 1.0
            sequences.append(single_batch)

        # stack sequences along batch dimension: [B, T, ...]
        batch_td = TensorDict.stack(sequences, 0)
        return batch_td

    def add(self, transition: Dict[str, torch.Tensor]) -> None:
        """
        Adds a transition to the replay buffer.

        Args:
            transition (Dict[str, torch.Tensor]): transition to add.
            All values must be tensors.
        """

        assert "is_first" in transition.keys(), "Must provide the 'is_first' key"
        if transition["is_first"] == 1.0:
            self._episode_id += 1
        transition["episode_id"] = torch.tensor([self._episode_id])
        self._buffer.add(TensorDict(transition, batch_size=[]))

        if len(self._buffer) % self._save_every == 0:
            print("Saving buffer....")
            self.save_compact(self._disk_path)
            # self._buffer.dumps(self._disk_path)

    def save_compact(self, path: str):
        length = len(self._buffer)
        td = self._buffer.storage[:length]
        torch.save(td, os.path.join(path, "buffer.pt"))

    def load_compact(self, path: str):
        path = os.path.join(path, "buffer.pt")
        if os.path.exists(path):
            print(f"Loading replay buffer at path: {path}")
            td = torch.load(path, weights_only=False)
            self._buffer.extend(td)

    def sample(
        self,
        batch_size: int,
        batch_length: int,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Samples a batch of sequential transitions from the replay buffer.

        Args:
            batch_size (int): size of the batch to sample

            batch_length (int): length of each batch in the sample

            device (torch.device, optional). Optional device to move the sampled tensors to.
            Defaults to None, in which case will be on CPU.

        Returns:
           sample of shape (batch_size, batch_length)
        """

        res = self._sample(batch_size, batch_length)
        if device is not None:
            res = res.to(device)
        return res
