#include "fileutils.h"

#include <assert.h>
#include <stdint.h>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#ifdef _WIN32
void* mmapFile(const char* path, size_t* outSize)
{
	*outSize = 0;

	HANDLE file = CreateFileA(path, GENERIC_READ,
	    FILE_SHARE_READ, // allow others to read
	    NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
	if (file == INVALID_HANDLE_VALUE)
		return NULL;

	LARGE_INTEGER liSize;
	if (!GetFileSizeEx(file, &liSize) || liSize.QuadPart == 0 || liSize.QuadPart > SIZE_MAX)
	{
		CloseHandle(file);
		return NULL;
	}

	HANDLE map = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
	if (!map)
	{
		CloseHandle(file);
		return NULL;
	}

	void* view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);

	// safe to close mapping handle & file handle after mapping the view; the kernel keeps track of the association via the mapped pointer
	CloseHandle(map);
	CloseHandle(file);

	if (!view)
		return NULL;

	*outSize = size_t(liSize.QuadPart);
	return view;
}

void unmapFile(void* data, size_t size)
{
	(void)size;

	BOOL ok = UnmapViewOfFile(data);
	assert(ok);
	(void)ok;
}
#else
void* mmapFile(const char* path, size_t* outSize)
{
	*outSize = 0;

	int fd = open(path, O_RDONLY);
	if (fd == -1)
		return NULL;

	struct stat sb;
	if (fstat(fd, &sb) == -1 || sb.st_size == 0 || sb.st_size > SIZE_MAX)
	{
		close(fd);
		return NULL;
	}

	void* mapped = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (mapped == MAP_FAILED)
	{
		close(fd);
		return NULL;
	}

	close(fd);

	*outSize = sb.st_size;
	return mapped;
}

void unmapFile(void* data, size_t size)
{
	int rc = munmap(data, size);
	assert(rc == 0);
	(void)rc;
}
#endif
