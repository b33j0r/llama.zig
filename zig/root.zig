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

    pub fn default() ModelLoader {
        return ModelLoader{
            .path = "local/Llama-3.2-3B-Instruct-Q5_K_M.gguf",
            // .path = "local/rocket-3b.Q5_0.gguf",
        };
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

    pub fn init(model: *Model, n_ctx: u32) !Context {
        var ctx_params = c.llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = n_ctx;

        const ctx = c.llama_new_context_with_model(model.ptr, ctx_params);
        if (ctx == null) return error.CreateContextFailed;

        return Context{
            .ptr = ctx,
            .model = model,
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
        // c code for reference:
        // llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        // llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
        // llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
        // llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        const params = c.llama_sampler_chain_default_params();
        const smpl = c.llama_sampler_chain_init(params);
        if (smpl == null) return error.CreateSamplerFailed;
        // c.llama_sampler_chain_add(smpl, c.llama_sampler_init_min_p(0.05, 1));
        // c.llama_sampler_chain_add(smpl, c.llama_sampler_init_temp(0.8));
        // c.llama_sampler_chain_add(smpl, c.llama_sampler_init_dist(c.LLAMA_DEFAULT_SEED));
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

pub const ResponseBuilder = struct {
    tokens: std.ArrayListUnmanaged(Token),
};

pub fn Pipeline(
    on_progress: ?fn (part: []const u8) void,
    on_complete: ?fn (response: []const u8) void,
) type {
    return struct {
        allocator: std.mem.Allocator,
        ctx: *Context,
        sampler: *Sampler,
        max_tokens: i32 = 1024,

        const Self = @This();

        pub fn generate(self: Self, prompt: []const u8) !void {
            const vocab = self.ctx.model.vocab;
            const n_prompt = -c.llama_tokenize(vocab, @as([*c]const u8, @ptrCast(prompt)), @as(i32, @intCast(prompt.len)), null, 0, true, true);
            if (n_prompt < 0) return error.TokenizationProbeFailed;

            const prompt_tokens = try std.heap.c_allocator.alloc(Token, @as(usize, @intCast(n_prompt)));
            defer std.heap.c_allocator.free(prompt_tokens);

            if (c.llama_tokenize(vocab, @as([*c]const u8, @ptrCast(prompt)), @as(i32, @intCast(prompt.len)), prompt_tokens.ptr, n_prompt, true, true) < 0) {
                return error.TokenizationFailed;
            }

            var batch = c.llama_batch_get_one(prompt_tokens.ptr, n_prompt);

            var n_pos: i32 = 0;
            var n_decode: i32 = 0;
            var new_token_id: c.llama_token = 0;

            // Pre-allocate response buffer for better memory management
            var response_buffer: std.ArrayListUnmanaged(u8) = .empty;
            defer response_buffer.deinit(self.allocator);

            // Reserve some space to reduce reallocations
            try response_buffer.ensureTotalCapacity(self.allocator, 4096);

            const max_length = n_prompt + self.max_tokens;
            while (n_pos + batch.n_tokens < max_length) {
                // evaluate the current batch with the transformer model
                if (c.llama_decode(self.ctx.ptr.?, batch) != 0) {
                    return error.EvalFailed;
                }

                n_pos += batch.n_tokens;

                // sample the next token
                new_token_id = self.sampler.sample(self.ctx);

                // is it an end of generation?
                if (c.llama_vocab_is_eog(vocab, new_token_id)) {
                    break;
                }

                var buf: [128]u8 = undefined;
                const n = c.llama_token_to_piece(vocab, new_token_id, &buf[0], buf.len, 0, true);
                if (n < 0) {
                    return error.TokenToPieceFailed;
                }

                const piece = buf[0..@as(usize, @intCast(n))];

                // Record current position in buffer before appending
                const start_pos = response_buffer.items.len;

                // Append the piece to our final buffer
                try response_buffer.appendSlice(self.allocator, piece);

                // Create slice from the buffer for the current part
                const part_slice = response_buffer.items[start_pos..];

                if (on_progress) |callback| {
                    callback(part_slice);
                }

                // update batch for next iteration
                batch = c.llama_batch_get_one(&new_token_id, 1);

                n_decode += 1;
            }

            if (on_complete) |callback| {
                // No need to allocate or join - we already have the complete response
                callback(response_buffer.items);
            }
        }
    };
}
