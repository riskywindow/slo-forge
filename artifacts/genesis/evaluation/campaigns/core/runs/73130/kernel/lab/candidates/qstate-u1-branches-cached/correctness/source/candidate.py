"""Generated untrusted HybridDecoder state-update candidate."""

import math

UNROLL_FACTOR = 1

def quantized_recurrent_state_update(
    previous_storage, activation_storage, count, previous_offset, previous_stride,
    activation_offset, activation_stride, output_alias_previous,
):
    if count < 1 or count > 128:
        raise ValueError('count outside supported domain')
    if previous_stride not in (1, 2, 3) or activation_stride not in (1, 2, 3):
        raise ValueError('stride outside supported domain')
    previous_coefficient = 0.625
    activation_coefficient = 31.0
    logical = [0] * count
    position = 0
    while position < count:
        stop = min(count, position + UNROLL_FACTOR)
        while position < stop:
            previous_value = previous_storage[previous_offset + position * previous_stride]
            activation = activation_storage[activation_offset + position * activation_stride]
            if previous_value < -127 or previous_value > 127:
                raise ValueError('previous state outside symmetric int8 domain')
            if not math.isfinite(activation):
                raise ValueError('activation must be finite')
            combined = previous_value * previous_coefficient + activation * activation_coefficient
            if math.isinf(combined):
                rounded = 127 if combined > 0 else -127
            else:
                rounded = round(combined)
            if rounded < -127:
                rounded = -127
            elif rounded > 127:
                rounded = 127
            logical[position] = rounded
            position += 1
    if output_alias_previous:
        for position in range(count):
            previous_storage[previous_offset + position * previous_stride] = logical[position]
    return logical
