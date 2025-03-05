void fill_array_float(float *array, size_t len, float val) {
    for (size_t i = 0; i < len; i++) {
        array[i] = val;
    }
}

void fill_array_float2(float2 *array, size_t len, float val) {
    for (size_t i = 0; i < len; i++) {
        array[i].x = val;
        array[i].y = val;
    }
}