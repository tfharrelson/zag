const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;

pub fn dot(
    comptime T: type,
    buffer: []T,
    left_offset: u64,
    right_offset: u64,
    vector_length: u64,
    output_index: u64,
) void {
    assert(buffer.len >= left_offset + vector_length);
    assert(buffer.len >= right_offset + vector_length);
    assert(output_index < buffer.len);
    buffer[output_index] = 0;
    for (0..vector_length) |i| {
        buffer[output_index] += buffer[left_offset + i] * buffer[right_offset + i];
    }
}

pub fn matmul(
    comptime T: type,
    buffer: []T,
    left_offset: u64,
    right_offset: u64,
    vector_length: u64,
    left_num_rows: u64,
    right_num_rows: u64,
    output_offset: u64,
) void {
    assert(right_offset >= left_offset + vector_length * left_num_rows);
    assert(output_offset >= right_offset + vector_length * right_num_rows);
    assert(buffer.len >= output_offset + left_num_rows * right_num_rows);
    // maybe do something more clever with threads?
    // need to profile to see if this is slow
    var output_index: u64 = output_offset;
    for (0..left_num_rows) |i| {
        for (0..right_num_rows) |j| {
            dot(
                T,
                buffer,
                left_offset + i * vector_length,
                right_offset + j * vector_length,
                vector_length,
                output_index,
            );
            output_index += 1;
        }
    }
}

pub fn add_inplace_vector(
    comptime T: type,
    buffer: []T,
    left_offset: u64,
    right_offset: u64,
    vector_length: u64,
) void {
    assert(buffer.len >= left_offset + vector_length);
    assert(buffer.len >= right_offset + vector_length);
    for (0..vector_length) |i| {
        buffer[left_offset + i] += buffer[right_offset + i];
    }
}

pub fn add_inplace_scalar(
    comptime T: type,
    buffer: []T,
    offset: u64,
    vector_length: u64,
    value: T,
) void {
    assert(buffer.len >= offset + vector_length);
    for (0..vector_length) |i| {
        buffer[offset + i] += value;
    }
}

pub fn mul_inplace_vector(
    comptime T: type,
    buffer: []T,
    left_offset: u64,
    right_offset: u64,
    vector_length: u64,
) void {
    assert(buffer.len >= left_offset + vector_length);
    assert(buffer.len >= right_offset + vector_length);
    for (0..vector_length) |i| {
        buffer[left_offset + i] *= buffer[right_offset + i];
    }
}

pub fn mul_inplace_scalar(
    comptime T: type,
    buffer: []T,
    offset: u64,
    vector_length: u64,
    value: T,
) void {
    assert(buffer.len >= offset + vector_length);
    for (0..vector_length) |i| {
        buffer[offset + i] *= value;
    }
}

pub fn copy_vector(
    comptime T: type,
    buffer: []T,
    vector_offset: u64,
    vector_length: u64,
    copy_offset: u64,
) void {
    assert(buffer.len >= vector_offset + vector_length);
    assert(buffer.len >= copy_offset + vector_length);
    for (0..vector_length) |i| {
        buffer[copy_offset + i] = buffer[vector_offset + i];
    }
}

pub fn apply_fn(
    comptime T: type,
    buffer: []T,
    offset: u64,
    vector_length: u64,
    func: *const fn (type, T) T,
) void {
    for (0..vector_length) |i| {
        buffer[offset + i] = func(buffer[offset + i]);
    }
}

pub fn relu(
    comptime T: type,
    value: T,
) T {
    return @max(value, 0);
}

// tests
test "dot" {
    var buffer = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0 };
    dot(f32, &buffer, 0, 4, 4, 8);
    expect(buffer[8] == 5.0 + 12.0 + 21.0 + 32.0) catch {
        std.debug.print("expected 5.0 + 12.0 + 21.0 + 32.0, got {d}\n", .{buffer[8]});
    };
    dot(f32, &buffer, 0, 1, 1, 8);
    try expect(buffer[8] == 2.0);
    dot(f32, &buffer, 0, 4, 2, 8);
    try expect(buffer[8] == 17.0);
}

test "add_inplace_vector" {
    var buffer = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    add_inplace_vector(f32, &buffer, 0, 4, 4);
    const expected = [_]f32{ 4.0, 6.0, 8.0, 10.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    for (buffer, expected) |a, b| {
        expect(a == b) catch {
            std.debug.print("expected buffer {any} to equal {any}\n", .{ buffer, expected });
            return error.TestFailed;
        };
    }
}

test "add_inplace_scalar" {
    var buffer = [_]f32{ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    add_inplace_scalar(f32, &buffer, 0, 4, 2.0);
    add_inplace_scalar(f32, &buffer, 4, 4, -10.0);
    const expected = [_]f32{ 2.0, 3.0, 4.0, 5.0, -6.0, -5.0, -4.0, -3.0, 8.0 };
    for (buffer, expected) |a, b| {
        expect(a == b) catch {
            std.debug.print("expected buffer {any} to equal {any}\n", .{ buffer, expected });
            return error.TestFailed;
        };
    }
}
