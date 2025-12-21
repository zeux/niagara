#include <stddef.h>

void* mmapFile(const char* path, size_t* outSize);
void unmapFile(void* data, size_t size);
