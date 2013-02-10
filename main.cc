
#include <math.h>
#include <SDL.h>
#include <fftw3.h>
#include "nanotime.h"


struct cell {
	float vx, vy;
	float mass;

	void clear() {
		mass = vx = vy = 0.0f;
	}
};

struct celld {
	struct cell d[2];
};

struct gravity_convolution {
	fftwf_complex* data;
	fftwf_complex* kernel;
	int exp;
	int n;
	size_t data_size;
	fftwf_plan data_forward_plan;
	fftwf_plan data_backward_plan;

	gravity_convolution(int exp) : exp(exp) {
		int mid = 1 << (exp-1);
		int size = 1 << exp;
		n = 1 << (exp+1);

		data_size = sizeof(fftwf_complex) * n * n;

		kernel = (fftwf_complex*) fftwf_malloc(data_size);

		fftwf_plan kernel_plan = fftwf_plan_dft_2d(n, n, kernel, kernel, -1, 0);
		for(int y = 0; y < n; y++) {
			for(int x = 0; x < n; x++) {
				if(x < size && y < size && !(x == mid && y == mid)) {
					float dx = mid - x;
					float dy = mid - y;
					float d2 = dx*dx + dy*dy;
					float di = 1.0f / sqrtf(d2);
					kernel_at(x, y)[0] = (dx * di) / d2;
					kernel_at(x, y)[1] = (dy * di) / d2;
				} else {
					// zero padding
					kernel_at(x, y)[0] = 0.0f;
					kernel_at(x, y)[1] = 0.0f;
				}
			}
		}

		fftwf_execute(kernel_plan);

		data = (fftwf_complex*) fftwf_malloc(data_size);
		data_forward_plan = fftwf_plan_dft_2d(n, n, data, data, -1, 0);
		data_backward_plan = fftwf_plan_dft_2d(n, n, data, data, 1, 0);

	}
	~gravity_convolution() {
		fftwf_free(data);
		fftwf_free(kernel);
	}

	void clear() {
		memset(data, 0, data_size);
	}

	fftwf_complex& data_at(int x, int y) const {
		return data[x + (y << (exp+1))];
	}

	fftwf_complex& post_data_at(int x, int y) const {
		int mid = 1 << (exp-1);
		return data[x + mid + ((y+mid) << (exp+1))];
	}

	fftwf_complex& kernel_at(int x, int y) const {
		return kernel[x + (y << (exp+1))];
	}

	void convolve() {
		uint64_t t0 = nanotime_get();
		fftwf_execute(data_forward_plan);
		int nn = n*n;
		for(int i = 0; i < nn; i++) {
			float new_re = data[i][0] * kernel[i][0] - data[i][1] * kernel[i][1];
			data[i][1] = data[i][1] * kernel[i][0] + data[i][0] * kernel[i][1];
			data[i][0] = new_re;
		}
		fftwf_execute(data_backward_plan);
		float dt = (nanotime_get() - t0) / 1e9f;
		printf("convolved in %fs\n", dt);
	}
};

struct automata {
	const unsigned int exp;
	struct celld* cells;
	struct cell empty;
	int d;
	float lost_mass;
	int time;
	struct gravity_convolution gc;

	const float MASS_EPSILON() const {
		// discard masses below this epsilon value
		return 0.0001;
	}

	automata(int exp) : exp(exp), d(0), gc(exp-1) {
		cells = (struct celld*) malloc(sizeof(struct celld) * area());
		lost_mass = 0.0f;
		empty.clear();
		time = 0;

	}

	void flipd() {
		d ^= 1;
	}

	~automata() {
		free(cells);
	}
	
	const int length() const {
		return 1<<exp;
	}

	const int area() const {
		return 1<<(exp<<1);
	}

	void cell_paint(struct cell& cell, int x0, int y0, int w, int h) {
		for(int y = y0; y < (y0+h); y++) {
			for(int x = x0; x < (x0+w); x++) {
				at(x,y) = cell;
			}
		}
	}

	void init_debug() {
		// clear
		cell_paint(empty, 0, 0, length(), length());

		// sphere of mass
		for(int y = 0; y < length(); y++) {
			for(int x = 0; x < length(); x++) {
				float dx = length()/2-x;
				float dy = length()/2-y;
				float d = dx * dx + dy * dy;
				if(d < 64000 && d > 1) {
					at(x,y).mass = 0.07;
				}
				if(d <= 1) {
					at(x,y).mass = 20.1f;
				}
			}
		}
	}

	struct celld& atc(int x, int y) const {
		return cells[x + y * length()];
	}

	struct cell& at(int x, int y) const {
		return atc(x,y).d[d];
	}

	struct cell& atz(int x, int y) const {
		return atc(x,y).d[d^1];
	}

	bool inside(int s) const {
		return s >= 0 && s < length();
	}

	void add_mass(int x, int y, float mass) {
		if(inside(x) && inside(y)) {
			at(x,y).mass += mass;
		}
	}

	void step() {
		uint64_t t0 = nanotime_get();

		// clear temporary buffer
		for(int y = 0; y < length(); y++) {
			for(int x = 0; x < length(); x++) {
				atz(x,y).clear();
			}
		}

		// do gravity once in a while
		if((time & 1) == 0) {
			gc.clear();
			// pack into half-resolution buffer
			int l2 = length() >> 1;
			for(int y = 0; y < l2; y++) {
				for(int x = 0; x < l2; x++) {
					gc.data_at(x,y)[0] =
						at((x<<1),(y<<1)).mass +
						at((x<<1)+1,(y<<1)).mass +
						at((x<<1),(y<<1)+1).mass +
						at((x<<1)+1,(y<<1)+1).mass;
					// vv complex mass! :-)
					//gc.data_at(x,y)[1] = gc.data_at(x,y)[0] * 0.05f;
				}
			}

			// calculate gravity
			gc.convolve();

			// unpack acceleration values
			for(int y = 0; y < l2; y++) {
				for(int x = 0; x < l2; x++) {
					float s = 0.00000001;
					float ax = gc.post_data_at(x,y)[0] * s;
					float ay = gc.post_data_at(x,y)[1] * s;

					at((x<<1),(y<<1)).vx += ax;
					at((x<<1),(y<<1)).vy += ay;

					at((x<<1)+1,(y<<1)).vx += ax;
					at((x<<1)+1,(y<<1)).vy += ay;

					at((x<<1),(y<<1)+1).vx += ax;
					at((x<<1),(y<<1)+1).vy += ay;

					at((x<<1)+1,(y<<1)+1).vx += ax;
					at((x<<1)+1,(y<<1)+1).vy += ay;
				}
			}
		}

		// apply pressure (perhaps do linear solve after this...?)
		float R = 0.18;
		//float R = 0.03;
		for(int y = 0; y < length(); y++) {
			for(int x = 0; x < length(); x++) {
				struct cell& o = at(x,y);
				for(int y0 = -1; y0 <= 1; y0++) {
					for(int x0 = -1; x0 <= 1; x0++) {
						if(x0 == 0 && y0 == 0) continue;
						if(inside(x+x0) && inside(y+y0)) {
							struct cell& p = at(x+x0,y+y0);
							if(o.mass != 0.0f || p.mass != 0.0f) {
								o.vx += (o.mass - p.mass) * R * x0;
								o.vy += (o.mass - p.mass) * R * y0;
								// TODO exchange mass as well?
							}
						}
					}
				}
			}
		}

		// advection. vx/vy are momentum (velocity * mass)
		for(int y = 0; y < length(); y++) {
			for(int x = 0; x < length(); x++) {
				struct cell& o = at(x,y);
				if(o.mass == 0.0f) continue;

				float nx = ((float)x) + o.vx;
				float ny = ((float)y) + o.vy;

				// XXX the 512 is to prevent ((int)-0.01) == 0 (-1 is the correct answer)
				int nx0 = ((int) (nx + 512.0f)) - 512;
				int nx1 = nx0 + 1;
				int ny0 = ((int) (ny + 512.0f)) - 512;
				int ny1 = ny0 + 1;

				float s1 = nx - nx0;
				float s0 = 1.0f - s1;
				float t1 = ny - ny0;
				float t0 = 1.0f - t1;

				if(inside(nx0)) {
					if(inside(ny0)) {
						float s = o.mass * s0 * t0;
						atz(nx0,ny0).mass += s;
						atz(nx0,ny0).vx += o.vx * s;
						atz(nx0,ny0).vy += o.vy * s;
					}
					if(inside(ny1)) {
						float s = o.mass * s0 * t1;
						atz(nx0,ny1).mass += s;
						atz(nx0,ny1).vx += o.vx * s;
						atz(nx0,ny1).vy += o.vy * s;
					}
				}
				if(inside(nx1)) {
					if(inside(ny0)) {
						float s = o.mass * s1 * t0;
						atz(nx1,ny0).mass += s;
						atz(nx1,ny0).vx += o.vx * s;
						atz(nx1,ny0).vy += o.vy * s;
					}
					if(inside(ny1)) {
						float s = o.mass * s1 * t1;
						atz(nx1,ny1).mass += s;
						atz(nx1,ny1).vx += o.vx * s;
						atz(nx1,ny1).vy += o.vy * s;
					}
				}
			}
		}

		// convert momentum back to velocity
		float rest_mass = 0.0f;
		float total_mass = 0.0f;
		for(int y = 0; y < length(); y++) {
			for(int x = 0; x < length(); x++) {
				struct cell& o = atz(x,y);
				if(o.mass > MASS_EPSILON()) {
					o.vx /= o.mass;
					o.vy /= o.mass;
					total_mass += o.mass;
				} else {
					rest_mass += o.mass;
					o.mass = 0.0f;
					o.vx = 0.0f;
					o.vy = 0.0f;
				}
			}
		}
		lost_mass += rest_mass;

		uint64_t dt = nanotime_get() - t0;
		printf("step: %llu\ttotal mass: %f\trest mass: %f\ttotal lost mass %f\n", dt, total_mass, rest_mass, lost_mass);

		flipd();
		time++;
	}

	void paint(SDL_Surface* screen) {
		SDL_LockSurface(screen);
		char* pixels = (char*) screen->pixels;
		for(int i = 0; i < area(); i++) {
			int m1 = cells[i].d[d].mass * 255.0f;
			int m2 = cells[i].d[d].mass * 50.0f;
			//int m3 = (cells[i].d[d].mass * cells[i].d[d].mass) * 6000.0f;
			pixels[i*screen->format->BytesPerPixel+1] = m1 > 255 ? 255 : m1;
			pixels[i*screen->format->BytesPerPixel+2] = m2 > 255 ? 255 : m2;
			//pixels[i*screen->format->BytesPerPixel+3] = m3 > 70 ? 70 : m3;
			pixels[i*screen->format->BytesPerPixel+3] = cells[i].d[d].mass != 0 ? 30 : 0;
		}
		SDL_UnlockSurface(screen);
	}
};

int main(int argc, char** argv) {
	SDL_Init(SDL_INIT_VIDEO);

	int exp = 9;
	int flags = 0;
	SDL_Surface* screen = SDL_SetVideoMode(1<<exp, 1<<exp, 32, flags);
	SDL_WM_SetCaption("OUT OF BEER", NULL);

	automata a(exp);

	a.init_debug();

	bool exiting = false;

	int mx = 0;
	int my = 0;
	bool paint = false;
	while(!exiting) {
		SDL_Event e;
		while(SDL_PollEvent(&e)) {
			switch(e.type) {
				case SDL_MOUSEMOTION:
					mx = e.motion.x;
					my = e.motion.y;
					break;
				case SDL_MOUSEBUTTONDOWN:
					paint = true;
					break;
				case SDL_MOUSEBUTTONUP:
					paint = false;
					break;
				case SDL_KEYDOWN:
					if(e.key.keysym.sym == SDLK_ESCAPE) exiting = true;
					break;
				case SDL_QUIT:
					exiting = true;
					break;
			}
		}
		if(paint) {
			float m = 4.0f;
			a.add_mass(mx-1, my-1, m);
			a.add_mass(mx, my-1, m);
			a.add_mass(mx+1, my-1, m);
			a.add_mass(mx-1, my, m);
			a.add_mass(mx, my, m);
			a.add_mass(mx+1, my, m);
			a.add_mass(mx-1, my+1, m);
			a.add_mass(mx, my+1, m);
			a.add_mass(mx+1, my+1, m);
		}
		a.step();
		a.paint(screen);
		SDL_Flip(screen);
	}

	return 0;
}

