#include <stddef.h>
#include <stdint.h>

void* mmapFile(const char* path, size_t* outSize);
void unmapFile(void* data, size_t size);

uint64_t hashFileMeta(const char* path);
