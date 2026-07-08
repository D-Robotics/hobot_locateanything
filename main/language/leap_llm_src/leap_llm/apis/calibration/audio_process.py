import torch
import torch.nn.functional as F
import librosa
from transformers import WhisperFeatureExtractor


def audio_attn_mask(seq_length, min_value, padded_mask_after_cnn, device):

    attention_mask = torch.full(
        [1, seq_length, seq_length],
        min_value,
        dtype=torch.float32,
    )
    cu_seqlens = torch.cat(
        (
            torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
            padded_mask_after_cnn.sum(1).cumsum(0),
        )
    ).to(torch.int32)

    for i in range(1, len(cu_seqlens)):
        attention_mask[
            ..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]
        ] = 0
    attention_mask = attention_mask.to(device)
    return attention_mask


class AudioChunkProcessor:
    """
    Processes a single audio file into padded feature chunks with masks.

    Example usage:
        processor = AudioChunkProcessor(
            device=device,
            n_window=100,
            feature_size=128,
            chunk_length=30,
            sampling_rate=16000
        )
        for audio_path in audio_paths:
            feat_list, mask_list, mask_cnn_list = processor(audio_path)
            for feat, mask, mask_cnn in zip(feat_list, mask_list, mask_cnn_list):
                # process each chunk
    """

    def __init__(
        self,
        device: torch.device,
        n_window: int = 100,
        feature_size: int = 128,
        chunk_length: int = 30,
        sampling_rate: int = 16000,
        attn_min_value: int = -512.0,
    ):
        self.device = device
        self.n_window = n_window
        self.sampling_rate = sampling_rate
        self.attn_min_value = attn_min_value

        # Initialize Whisper feature extractor
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=feature_size, chunk_length=chunk_length
        )

        # Default kwargs for extractor
        self.output_kwargs = {
            "sampling_rate": sampling_rate,
            "padding": "max_length",
            "return_attention_mask": True,
            "return_tensors": "pt",
        }

    def _padded_and_mask(
        self, tensor_list, tensor_len, padding_value=0, padding_side="right"
    ):
        """
        Pads a list of feature tensors and returns:
          - padded feature tensor [batch, dim, max_len]
          - attention mask before CNN [batch, 1, max_len]
          - mask after CNN pooling [batch, max_len_after_cnn]
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            (len(tensor_list), dim, max_len),
            padding_value,
            dtype=torch.float32,
            device=tensor_list[0].device,
        )
        batch_mask = torch.zeros(
            (len(tensor_list), max_len), dtype=torch.long, device=padded_tensor.device
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        # After a CNN with stride=2
        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_list), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1

        return (padded_tensor, batch_mask.unsqueeze(1), batch_mask_after_cnn.bool())

    def get_feature(self, audio_path: str):
        # Load and extract features
        audio_signal = librosa.load(audio_path, sr=self.sampling_rate)[0]
        waveform = torch.from_numpy(audio_signal).float()
        audio_inputs = self.feature_extractor(waveform, **self.output_kwargs)

        # External helper to obtain features and lengths
        input_features, feature_lens = get_audio_input_feature(audio_inputs)
        return input_features, feature_lens

    def __call__(self, input_features, feature_lens):
        input_features = input_features.to(self.device)
        # Split into chunks of size n_window*2
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        total_chunks = chunk_num.sum().item()
        chunk_lengths = torch.full(
            (total_chunks,),
            self.n_window * 2,
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_indices = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_indices] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(
            chunk_lengths == 0, self.n_window * 2, chunk_lengths
        )
        chunks = input_features.split(chunk_lengths.tolist(), dim=1)

        # Pad and mask
        padded_feat, padded_mask, padded_mask_cnn = self._padded_and_mask(
            chunks, chunk_lengths, padding_value=0, padding_side="right"
        )

        # Split batched tensors to lists
        feat_list = padded_feat.split(1, dim=0)
        mask_list = padded_mask.split(1, dim=0)
        mask_cnn_list = padded_mask_cnn.split(1, dim=0)
        attention_mask_list = [
            audio_attn_mask(
                seq_length=self.n_window,
                min_value=self.attn_min_value,
                padded_mask_after_cnn=mask_cnn,
                device=self.device,
            )
            for mask_cnn in mask_cnn_list
        ]

        return feat_list, mask_list, attention_mask_list


class Qwen2AudioPreprocessor:
    """
    # audio_processor = Qwen2AudioPreprocessor()
    # input_features = audio_processor.feature_extractor(auido_path)[0]
    # print("=== input_features:", input_features.shape)
    # feature_lens = torch.tensor([input_features.shape[-1]])

    """

    def __init__(self):
        super().__init__()
        self.audio_embeds = None
        self.audio_pad_id = 151646
        self.n_fft = 400
        self.sampling_rate = 16000
        self.hop_length = 160
        self.chunk_length = 30
        self.feature_size = 128
        self.n_samples = self.chunk_length * self.sampling_rate
        self.max_length = self.n_samples // self.hop_length
        from transformers.audio_utils import mel_filter_bank

        self.mel_filters = mel_filter_bank(
            num_frequency_bins=1 + self.n_fft // 2,
            num_mel_filters=self.feature_size,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=self.sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )

    def _torch_extract_fbank_features(self, waveform):
        window = torch.hann_window(self.n_fft)
        stft = torch.stft(
            waveform, self.n_fft, self.hop_length, window=window, return_complex=True
        )
        magnitudes = stft[..., :-1].abs() ** 2
        mel_filters = torch.from_numpy(self.mel_filters).type(torch.float32)
        mel_spec = mel_filters.T @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        if waveform.dim() == 2:
            max_val = log_spec.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            log_spec = torch.maximum(log_spec, max_val - 8.0)
        else:
            log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def feature_extractor(self, audio_content):
        audio_obj = librosa.load(audio_content, sr=self.sampling_rate)[0]
        waveform = torch.from_numpy(audio_obj).type(torch.float32)
        input_features = self._torch_extract_fbank_features(waveform).unsqueeze(0)
        return input_features


def get_audio_input_feature(audio_inputs):
    input_features = audio_inputs["input_features"]
    audio_inputs["feature_attention_mask"] = audio_inputs.pop(
        "attention_mask"
    )  # rename feature_attention_mask to prevent conflicts later on
    audio_inputs["input_features"] = audio_inputs.pop(
        "input_features"
    )  # rename input_features to prevent conflicts later on
    feature_attention_mask = audio_inputs["feature_attention_mask"]
    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        input_features = input_features.permute(0, 2, 1)[
            feature_attention_mask.bool()
        ].permute(1, 0)
    else:
        audio_feature_lengths = None

    feature_lens = (
        audio_feature_lengths
        if audio_feature_lengths is not None
        else feature_attention_mask.sum(-1)
    )

    return input_features, feature_lens
