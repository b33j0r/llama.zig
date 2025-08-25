const std = @import("std");

pub const c = @cImport({
    @cDefine("LLAMA_STATIC", "1");
    @cInclude("llama.h");
    @cInclude("ggml.h");
});

/// Top-level interface to the llama.cpp bindings.
pub const Llama = struct {
    pub fn init() Llama {
        c.llama_backend_init();
        c.ggml_backend_load_all();
        return Llama{};
    }

    pub fn deinit(_: Llama) void {
        c.llama_backend_free();
    }
};

pub const ModelLoader = struct {
    path: [:0]const u8,

    pub const default_model_path = "local/Llama-3.2-3B-Instruct-Q5_K_M.gguf";
    // pub const default_model_path = "local/rocket-3b.Q5_0.gguf";

    pub fn fromPath(path: [:0]const u8) ModelLoader {
        return ModelLoader{
            .path = path,
        };
    }

    pub fn default() ModelLoader {
        return ModelLoader.fromPath(default_model_path);
    }

    pub fn load(self: ModelLoader) !Model {
        const params = c.llama_model_default_params();

        const m = c.llama_load_model_from_file(@as([*c]const u8, @ptrCast(self.path)), params);
        if (m == null) return error.LoadModelFailed;

        return try Model.init(m.?);
    }
};

pub const Model = struct {
    ptr: *c.llama_model,
    vocab: *const c.llama_vocab,

    pub fn init(m: *c.llama_model) !Model {
        const vocab = c.llama_model_get_vocab(m);
        if (vocab == null) {
            c.llama_free_model(m);
            return error.NoVocab;
        }

        return Model{
            .ptr = m,
            .vocab = vocab.?,
        };
    }

    pub fn deinit(self: *Model) void {
        c.llama_free_model(self.ptr);
    }
};

pub const Context = struct {
    ptr: ?*c.llama_context,
    model: *Model,
    params: Params,

    pub const Params = struct {
        n_ctx: u32 = 8 * 1024,
        n_batch: u32 = 8 * 1024,
        max_tokens: i32 = 1024,
    };

    pub fn init(model: *Model, params: Params) !Context {
        var ctx_params = c.llama_context_default_params();
        ctx_params.n_ctx = params.n_ctx;
        ctx_params.n_batch = params.n_batch;

        const ctx = c.llama_new_context_with_model(model.ptr, ctx_params);
        if (ctx == null) return error.CreateContextFailed;

        return Context{
            .ptr = ctx,
            .model = model,
            .params = params,
        };
    }

    pub fn deinit(self: *Context) void {
        std.debug.assert(self.ptr != null);
        c.llama_free(self.ptr.?);
        self.ptr = null;
    }
};

pub const Sampler = struct {
    ptr: *c.llama_sampler,

    pub fn default() !Sampler {
        const params = c.llama_sampler_chain_default_params();
        const smpl = c.llama_sampler_chain_init(params);
        if (smpl == null) return error.CreateSamplerFailed;
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_min_p(0.05, 1));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_temp(0.85));
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_dist(c.LLAMA_DEFAULT_SEED));

        return Sampler{
            .ptr = smpl,
        };
    }

    pub fn greedy() !Sampler {
        const params = c.llama_sampler_chain_default_params();
        const smpl = c.llama_sampler_chain_init(params);
        if (smpl == null) return error.CreateSamplerFailed;
        c.llama_sampler_chain_add(smpl, c.llama_sampler_init_greedy());

        return Sampler{
            .ptr = smpl,
        };
    }

    pub fn deinit(self: *Sampler) void {
        c.llama_sampler_free(self.ptr);
    }

    pub fn sample(self: *Sampler, ctx: *const Context) c.llama_token {
        return c.llama_sampler_sample(self.ptr, ctx.ptr.?, -1);
    }
};

pub const Token = c.llama_token;

// New iterator that yields pieces synchronously and owns prompt token memory
pub const PipelineIterator = struct {
    allocator: std.mem.Allocator,
    ctx: *Context,
    sampler: *Sampler,
    vocab: *const c.llama_vocab,

    // prompt tokenization
    prompt_tokens: []Token,
    n_prompt: i32,

    // decoding state
    batch: c.llama_batch,
    n_pos: i32,
    n_decode: i32,
    max_tokens: i32,
    max_length: i32,
    new_token_id: c.llama_token,
    done: bool,

    // accumulated response
    response_buffer: std.ArrayListUnmanaged(u8),

    pub fn deinit(self: *PipelineIterator) void {
        if (self.prompt_tokens.len != 0) {
            std.heap.c_allocator.free(self.prompt_tokens);
            self.prompt_tokens = &[_]Token{};
        }
        self.response_buffer.deinit(self.allocator);
    }

    // Returns the next generated piece or null when finished.
    pub fn next(self: *PipelineIterator) !?[]const u8 {
        if (self.done) return null;

        if (self.n_pos + self.batch.n_tokens >= self.max_length) {
            self.done = true;
            return null;
        }

        // Evaluate current batch
        if (c.llama_decode(self.ctx.ptr.?, self.batch) != 0) {
            return error.EvalFailed;
        }
        self.n_pos += self.batch.n_tokens;

        // Sample next token
        self.new_token_id = self.sampler.sample(self.ctx);

        // End of generation?
        if (c.llama_vocab_is_eog(self.vocab, self.new_token_id)) {
            self.done = true;
            return null;
        }

        // Convert token to piece
        var buf: [128]u8 = undefined;
        const n = c.llama_token_to_piece(self.vocab, self.new_token_id, &buf[0], buf.len, 0, true);
        if (n < 0) return error.TokenToPieceFailed;
        const piece = buf[0..@as(usize, @intCast(n))];

        // Append to response buffer and return just the newly appended slice
        const start_pos = self.response_buffer.items.len;
        try self.response_buffer.appendSlice(self.allocator, piece);
        const part_slice = self.response_buffer.items[start_pos..];

        // Prepare batch for next iteration
        self.batch = c.llama_batch_get_one(&self.new_token_id, 1);
        self.n_decode += 1;

        return part_slice;
    }

    /// Like `next()` but returns null for errors too.
    pub fn nextIgnoreErrors(self: *PipelineIterator) ?[]const u8 {
        return self.next() catch null;
    }

    // Returns the accumulated full text so far
    pub fn full(self: *const PipelineIterator) []const u8 {
        return self.response_buffer.items;
    }
};

pub const Pipeline = struct {
    allocator: std.mem.Allocator,
    ctx: *Context,
    sampler: *Sampler,

    const Self = @This();

    pub fn generate(self: Self, prompt: []const u8) !PipelineIterator {
        const vocab = self.ctx.model.vocab;

        // Probe to get tokenization length
        const n_prompt = -c.llama_tokenize(
            vocab,
            @as([*c]const u8, @ptrCast(prompt)),
            @as(i32, @intCast(prompt.len)),
            null,
            0,
            true,
            true,
        );
        if (n_prompt < 0) return error.TokenizationProbeFailed;

        // Allocate prompt tokens (owned by the iterator)
        const prompt_tokens = try std.heap.c_allocator.alloc(Token, @as(usize, @intCast(n_prompt)));
        errdefer std.heap.c_allocator.free(prompt_tokens);

        if (c.llama_tokenize(
            vocab,
            @as([*c]const u8, @ptrCast(prompt)),
            @as(i32, @intCast(prompt.len)),
            prompt_tokens.ptr,
            n_prompt,
            true,
            true,
        ) < 0) {
            return error.TokenizationFailed;
        }

        // Prepare initial batch
        const batch = c.llama_batch_get_one(prompt_tokens.ptr, n_prompt);

        // Initialize iterator
        var it = PipelineIterator{
            .allocator = self.allocator,
            .ctx = self.ctx,
            .sampler = self.sampler,
            .vocab = vocab,

            .prompt_tokens = prompt_tokens,
            .n_prompt = n_prompt,

            .batch = batch,
            .n_pos = 0,
            .n_decode = 0,
            .max_tokens = self.ctx.params.max_tokens,
            .max_length = n_prompt + self.ctx.params.max_tokens,
            .new_token_id = 0,
            .done = false,

            .response_buffer = .empty,
        };

        // Pre-reserve for fewer reallocations
        try it.response_buffer.ensureTotalCapacity(self.allocator, 4096);

        return it;
    }
};
