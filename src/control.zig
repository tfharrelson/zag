const std = @import("std");
const assert = std.debug.assert;
const core = @import("core.zig");
const expect = std.testing.expect;

pub fn State(
    comptime T: type,
) type {
    return struct {
        const Self = @This();

        buffer: []T = undefined,
        input_offset: u64 = undefined,
        model_offset: u64 = undefined,
        intermediate_offset: u64 = undefined,

        pub fn init(buffer: []T, input_size: u64, model_size: u64) Self {
            return Self{
                .buffer = buffer,
                .input_offset = 0,
                .model_offset = input_size,
                .intermediate_offset = input_size + model_size,
            };
        }
    };
}

pub const Config = struct {
    task_bytes: u64,
};

pub fn Linear(
    comptime T: type,
    comptime in_layers: u64,
    comptime out_layers: u64,
) type {
    return struct {
        in_layers: u64 = in_layers,
        out_layers: u64 = out_layers,
        batch_size: u64 = undefined,
        curr_offset: u64 = 0,
        curr_task: u64 = 0,

        const Self = @This();

        pub fn init(comptime config: Config) Self {
            const batch_size: u64 = @divFloor(config.task_bytes, in_layers);
            if (batch_size == 0) {
                var buf: [64]u8 = undefined;
                const s = try std.fmt.bufPrint(
                    &buf,
                    "batch size must be greater than 0, task size: {d}, in layers: {d}\n",
                    .{ config.task_bytes, in_layers },
                );
                @compileError(s);
            }
            return Self{
                .in_layers = in_layers,
                .out_layers = out_layers,
                .batch_size = batch_size,
                .curr_offset = 0,
                .curr_task = 0,
            };
        }
        // TODO: generalize this to any type of state (include gpus)
        pub fn forward(self: *Self, state: *State(T), input_length: u64) bool {
            // get all the tasks required to execute this
            if (self.curr_offset >= input_length) {
                return false;
            }
            var task_num_rows: u64 = self.batch_size;
            if (self.curr_offset + self.batch_size > input_length) {
                task_num_rows = @divFloor(input_length - self.curr_offset, self.in_layers);
            }
            // use core functions to do the work
            const output_offset = state.intermediate_offset + self.curr_task * task_num_rows * self.out_layers;
            core.matmul(
                T,
                state.buffer,
                self.curr_offset,
                state.model_offset,
                self.in_layers,
                task_num_rows,
                self.out_layers,
                output_offset,
            );
            core.add_inplace_vector(
                T,
                state.buffer,
                output_offset,
                // TODO: figure out a better way to register model component offsets
                state.model_offset + self.in_layers * self.out_layers,
                task_num_rows * self.out_layers,
            );
            self.curr_offset += task_num_rows * self.in_layers;
            self.curr_task += 1;
            return true;
        }
    };
}

test "linear forward" {
    // TODO: a little tedious to set up the buffer
    var buffer = [_]f32{ 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var state = State(f32).init(&buffer, 3, 8);
    var linear = Linear(f32, 3, 2).init(.{ .task_bytes = 4 });
    var check = linear.forward(&state, 3);
    try expect(check == true);
    check = linear.forward(&state, 3);
    try expect(check == false);
    const expected = [_]f32{ 9.0, 27.0 };
    for (0..expected.len) |i| {
        try expect(buffer[state.intermediate_offset + i] == expected[i]);
    }
}
