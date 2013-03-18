#include <stdlib.h>
#include <fftw3.h>
#include "nanotime.h"

struct scope_timer {
	const char* timee;
	uint64_t t0;
	scope_timer(const char* timee) : timee(timee) {
		t0 = nanotime_get();
	}
	~scope_timer() {
		uint64_t dt = nanotime_get() - t0;
		float s = ((float)dt)/1e6;
		printf("%s: %f ms\n", timee, s);
	}
};


int main(int argc, char** argv) {
	if(argc != 3 && argc != 4) {
		fprintf(stderr, "usage: %s <width exp> <height exp> [n]\n", argv[0]);
		return 1;
	}

	int width = 1<<atoi(argv[1]);
	int height = 1<<atoi(argv[2]);

	int n = 1;
	if(argc == 4) {
		n = atoi(argv[3]);
	}

	printf("FFT for %dx%d\n", width, height);

	fftwf_plan data_forward_plan;
	fftwf_plan data_backward_plan;

	size_t data_size = sizeof(fftwf_complex) * width * height;
	fftwf_complex* data = (fftwf_complex*) fftwf_malloc(data_size);

	for(int i = 0; i < (width*height); i++) {
		data[i][0] = 1.1f;
		data[i][1] = 2.2f;
	}

	data_forward_plan = fftwf_plan_dft_2d(width, height, data, data, -1, 0);
	data_backward_plan = fftwf_plan_dft_2d(width, height, data, data, 1, 0);

	uint64_t t0 = nanotime_get();
	for(int i = 0; i < n; i++) {
		fftwf_execute(data_forward_plan);
		fftwf_execute(data_backward_plan);
	}
	uint64_t dt = nanotime_get() - t0;
	float ms = ((float)dt)/1e6;
	printf("%d X FFT: %f ms total / %f ms per 2xFFT / %f million cells per second\n", 2*n, ms, ms / n, (2000.0f * n * width * height) / (ms * 1e6));

	return 0;
}

