/* Syllabus: metadata for every topic. Content modules are loaded lazily. */

export const tracks = {
  dl:   { id: 'dl',   name: 'Deep Learning',   sub: 'From convolutions to modern LLMs',      acc: '#8b7bff', acc2: '#22d3ee' },
  cuda: { id: 'cuda', name: 'GPU Programming', sub: 'CUDA from warps to Tensor Cores',       acc: '#76d12a', acc2: '#22d3ee' }
}

const mods = import.meta.glob('./*/*/index.jsx')
const L = d => {
  const f = () => (mods[`./${d}/index.jsx`] ? mods[`./${d}/index.jsx`]() : Promise.reject(new Error('Chapter not written yet: ' + d)))
  f.ready = !!mods[`./${d}/index.jsx`]   // false for chapters that are registered but not authored yet
  return f
}

const t = (track, order, id, title, blurb, extra, load) => ({ track, order, id, title, blurb, ...extra, load, ready: !!load.ready })

export const topics = [
  t('dl', 1, 'cnn', 'Convolutional Networks',
    'Local receptive fields, weight sharing, stride/padding/dilation, pooling, receptive-field growth and the ResNet idea.',
    { level: 'Intermediate', time: '35 min', tags: ['conv', 'pooling', 'ResNet'] }, L('dl/cnn')),
  t('dl', 2, 'optim', 'Optimization & Loss Landscapes',
    'SGD, momentum, RMSProp and Adam racing across a 3D loss surface. Why training is geometry.',
    { level: 'Intermediate', time: '30 min', tags: ['SGD', 'Adam', 'schedules'] }, L('dl/optim')),
  t('dl', 3, 'rnn', 'Recurrent Neural Networks',
    'Hidden state as memory, backprop through time, and why gradients vanish or explode.',
    { level: 'Intermediate', time: '30 min', tags: ['BPTT', 'vanishing gradients'] }, L('dl/rnn')),
  t('dl', 4, 'lstm', 'LSTM',
    'The cell-state highway and the forget / input / output gates that fix long-range memory.',
    { level: 'Intermediate', time: '30 min', tags: ['gates', 'cell state'] }, L('dl/lstm')),
  t('dl', 5, 'gru', 'GRU',
    'The leaner gated cell: update and reset gates, parameter economics, and when to prefer it.',
    { level: 'Intermediate', time: '20 min', tags: ['gates', 'efficiency'] }, L('dl/gru')),
  t('dl', 6, 'attention', 'Seq2Seq & Attention',
    'Breaking the fixed-vector bottleneck: encoder–decoder models and Bahdanau / Luong attention.',
    { level: 'Intermediate', time: '30 min', tags: ['seq2seq', 'alignment'] }, L('dl/attention')),
  t('dl', 7, 'transformer', 'The Transformer',
    'Scaled dot-product attention, multi-head attention, positional encodings and the full block.',
    { level: 'Advanced', time: '50 min', tags: ['self-attention', 'multi-head', 'positional'] }, L('dl/transformer')),
  t('dl', 8, 'bert', 'BERT & Encoders',
    'Masked language modeling, bidirectional context, pre-train / fine-tune and the encoder family.',
    { level: 'Advanced', time: '35 min', tags: ['MLM', 'fine-tuning'] }, L('dl/bert')),
  t('dl', 9, 'gpt', 'GPT & Autoregressive LLMs',
    'Causal decoding, sampling strategies (temperature, top-k, top-p), KV cache and inference cost.',
    { level: 'Advanced', time: '45 min', tags: ['decoding', 'KV cache', 'inference'] }, L('dl/gpt')),
  t('dl', 10, 'modern', 'Modern LLM Toolkit',
    'RoPE, GQA, RMSNorm/SwiGLU, Mixture-of-Experts, LoRA/QLoRA, scaling laws and alignment (RLHF, DPO).',
    { level: 'Advanced', time: '60 min', tags: ['RoPE', 'MoE', 'LoRA', 'scaling laws'] }, L('dl/modern')),

  t('cuda', 1, 'cuda-exec', 'Execution Model: Grid, Block, Warp',
    'Walk the thread hierarchy in 3D, derive global indices, and watch blocks get scheduled onto SMs.',
    { level: 'Intermediate', time: '40 min', tags: ['grid', 'block', 'warp', 'SM'] }, L('cuda/exec')),
  t('cuda', 2, 'cuda-memory', 'The Memory Hierarchy & Roofline',
    'Registers, shared memory, L2, HBM: latencies, bandwidth and the roofline model that tells you what to optimize.',
    { level: 'Intermediate', time: '50 min', tags: ['registers', 'HBM', 'roofline'] }, L('cuda/memory')),
  t('cuda', 3, 'cuda-tiled', 'Shared Memory & Tiled MatMul',
    'Step through tiled matrix multiply: cooperative loads, __syncthreads, and 16x less global traffic.',
    { level: 'Advanced', time: '50 min', tags: ['tiling', 'shared memory', 'GEMM'] }, L('cuda/tiled')),
  t('cuda', 4, 'cuda-coalesce', 'Coalescing & Bank Conflicts',
    'Count real memory transactions and shared-memory bank conflicts for any access pattern.',
    { level: 'Advanced', time: '45 min', tags: ['coalescing', 'banks', 'padding'] }, L('cuda/coalesce')),
  t('cuda', 5, 'cuda-warp', 'Warps: Divergence, Shuffles, Reduction',
    'SIMT execution, divergence masks, warp-level primitives and the fastest way to reduce.',
    { level: 'Advanced', time: '50 min', tags: ['SIMT', 'divergence', 'shuffle'] }, L('cuda/warp')),
  t('cuda', 6, 'cuda-occupancy', 'Occupancy & Latency Hiding',
    'Registers, shared memory and block size fight over SM resources. Compute occupancy and see the limiter.',
    { level: 'Advanced', time: '45 min', tags: ['occupancy', 'registers'] }, L('cuda/occupancy')),
  t('cuda', 7, 'cuda-streams', 'Streams & Async Overlap',
    'Pipeline copy and compute with streams and pinned memory; simulate the timeline.',
    { level: 'Advanced', time: '45 min', tags: ['streams', 'pinned', 'overlap'] }, L('cuda/streams')),
  t('cuda', 8, 'cuda-flash', 'Tensor Cores & FlashAttention',
    'Matrix-multiply hardware, mixed precision, and how tiling + online softmax makes attention IO-aware.',
    { level: 'Expert', time: '60 min', tags: ['WMMA', 'FlashAttention', 'FP16/FP8'] }, L('cuda/flash'))
]

export const byTrack = tr => topics.filter(x => x.track === tr).sort((a, b) => a.order - b.order)
export const ordered = () => [...byTrack('dl'), ...byTrack('cuda')]
export const find = id => topics.find(x => x.id === id)
