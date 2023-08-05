
#include <math.h>
#include <SDL.h>
#include <fftw3.h>
#include "nanotime.h"
#include <vector>
#include <assert.h>

#include "video_out.h"

#define PI (3.14159265359f)
#define HALFPI (PI*0.5f)
#define PI2 (PI*2.0f)


struct scope_timer {
	const char* timee;
	uint64_t t0;
	scope_timer(const char* timee) : timee(timee) {
		t0 = nanotime_get();
	}
	~scope_timer() {
		//uint64_t dt = nanotime_get() - t0;
		//printf("%s: %llu ns\n", timee, dt);
	}
};

template <class T>
T lerp(T a, T b, T x) {
	return a + (b-a) * x;
}

struct vec2 {
	float u,v;
	vec2() : u(0), v(0) {}
	vec2(float u, float v) : u(u), v(v) {}

	struct vec2& operator+=(const struct vec2& other) {
		u += other.u;
		v += other.v;
		return *this;
	}
	struct vec2& operator-=(const struct vec2& other) {
		u -= other.u;
		v -= other.v;
		return *this;
	}
	struct vec2& operator*=(const float scalar) {
		u *= scalar;
		v *= scalar;
		return *this;
	}
	struct vec2 operator+(const struct vec2& other) const {
		return vec2(u+other.u, v+other.v);
	}
	struct vec2 operator-(const struct vec2& other) const {
		return vec2(u-other.u, v-other.v);
	}
	struct vec2 operator*(float scalar) const {
		return vec2(u*scalar, v*scalar);
	}
	struct vec2 operator/(float scalar) const {
		return vec2(u/scalar, v/scalar);
	}

	float dot(const struct vec2& other) const {
		return u * other.u + v * other.v;
	}

	float cross(const struct vec2& other) const {
		return u * other.v - v * other.u;
	}

	struct vec2 normal() const {
		return vec2(-v, u);
	}
};


struct solid_cell {
	float mass;
};

struct cell {
	float vx, vy;
	float mass;
	float XXX_residual;

	void clear() {
		XXX_residual = mass = vx = vy = 0.0f;
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
		scope_timer tt("convolution");

		fftwf_execute(data_forward_plan);
		int nn = n*n;
		for(int i = 0; i < nn; i++) {
			float new_re = data[i][0] * kernel[i][0] - data[i][1] * kernel[i][1];
			data[i][1]   = data[i][1] * kernel[i][0] + data[i][0] * kernel[i][1];
			data[i][0]   = new_re;
		}
		fftwf_execute(data_backward_plan);
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
	std::vector<struct solid*> solids;
	uint8_t* bitmap;

	const float MASS_EPSILON() const {
		// discard masses below this epsilon value
		return 0.0001;
	}

	automata(int exp) : exp(exp), d(0), gc(exp-1) {
		cells = (struct celld*) malloc(sizeof(struct celld) * area());
		lost_mass = 0.0f;
		empty.clear();
		time = 0;
		bitmap = (uint8_t*) malloc((1<<exp) * (1<<exp) * 4);

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
				#if 0
				if (d < 54000 && d > 5000) {
					//at(x,y).mass = 0.47/2;
					//at(x,y).vx = dy * 0.0007f + dx * 0.001f;
					//at(x,y).vy = -dx * 0.0007f + dy * 0.001f;

					//const float sx = 1.0007f;
					//const float sy = 0.001;
					//at(x,y).vx = dy * sx + dx * sy;
					//at(x,y).vy = -dx * sx + dy * sy;

					at(x,y).mass = 0.09f;
				}
				if (d <= 1000) {
					at(x,y).mass = 0.1f;
				}
				#endif
				//if (1000 < d && d < 74000) {

				/*
				float sd = sin(d*0.0006);
				if (d < 60000 && sd > 0) {
					at(x,y).mass = 0.255f * sd;
				}
				*/

				/*

				const float ss = 0.06f;
				const float mx = dx * ss;
				const float my = dy * ss;
				const float dd = fabsf(mx*tanf(sqrtf(mx*mx + my*my)) - my);
				if (dd < 2.0f) {
					at(x,y).mass = dd * 0.6f;
				}


				//at(x,y).mass = 0.05f;

				const float s = 2.1e-3f;
				const float t = -1.4e-3f;
				//const float s = 0;
				//const float t = 0;
				at(x,y).vx = dy * s + dx * t;
				at(x,y).vy = -dx * s + dy * t;
				*/

				/*
				if (d < 2000) {
					at(x,y).mass = 1.2f;
					const float f = 0.01f;
					at(x,y).vx = -dx * f;
					at(x,y).vy = -dy * f;
				} else if (40000 < d && d < 70000) {
					at(x,y).mass = 0.02f;
					const float f = 0.02f;
					at(x,y).vx = dx * f;
					at(x,y).vy = dy * f;
				}
				*/

				if (d < 1000) at(x,y).mass = 0.8f;
				if (20000 < d && d < 30000) at(x,y).mass = 0.1f;




			}
		}
		// XXX TODO clean up!
		#if 0
		struct solid* solid = new struct solid(280, 120);
		solid->position = vec2(256, 256);
		solid->rotation = 0.0f;
		solid->init_debug();
		solids.push_back(solid);
		#endif
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
		scope_timer tt("step");

		// clear temporary buffer
		for(int y = 0; y < length(); y++) {
			for(int x = 0; x < length(); x++) {
				atz(x,y).clear();
				//at(x,y).clear(); // XXX
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
					//const float complex_mass = 0.1f;
					const float complex_mass = 0.00f;
					gc.data_at(x,y)[1] = gc.data_at(x,y)[0] * complex_mass;
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
								//if(o.mass < 0.5 && p.mass < 0.5) continue;
								float f = (o.mass*o.mass - p.mass*p.mass); 
								//f = f*f;
								o.vx += f * R * x0;
								o.vy += f * R * y0;
								// TODO exchange mass as well?
							}
						}
					}
				}
			}
		}
		/*
		for(int y = 0; y < length(); y++) {
			for(int x = 0; x < length(); x++) {
				at(x,y).mass = atz(x,y).mass;
				atz(x,y).mass = 0.0f;
			}
		}
		*/

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

		//printf("total mass: %f\trest mass: %f\ttotal lost mass %f\n", total_mass, rest_mass, lost_mass);

		flipd();
		time++;
	}

	void paint() {
		for(int i = 0; i < area(); i++) {
			int m1 = cells[i].d[d].mass * 500.0f;
			int m2 = cells[i].d[d].mass * 150.0f;
			int m3 = (cells[i].d[d].mass * cells[i].d[d].mass + cells[i].d[d].XXX_residual) * 100.0f;
			bitmap[i*4+3] = 255;
			bitmap[i*4+2] = m1 > 255 ? 255 : m1;
			bitmap[i*4+1] = m2 > 255 ? 255 : m2;
			bitmap[i*4+0] = cells[i].d[d].mass != 0 ? (m3 < 30 ? 30 : (m3 > 255 ? 255 : m3)) : 0;
			//pixels[i*screen->format->BytesPerPixel+3] = m3 > 70 ? 70 : m3;
			//pixels[i*screen->format->BytesPerPixel+3] = cells[i].d[d].mass != 0 ? 30 : 0;
		}
	}
};

int main(int argc, char** argv) {
	SDL_Init(SDL_INIT_VIDEO);

	const int exp = 9;
	const int size = 1<<exp;

	int bitmask = SDL_WINDOW_OPENGL;
	SDL_Window* window = SDL_CreateWindow(
		"OUT OF BEER",
		SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
		size*2, size*2,
		bitmask);

	SDL_Renderer* r = SDL_CreateRenderer(window, -1, 0);
	SDL_Texture* t = SDL_CreateTexture(r, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, size, size);

	automata a(exp);

	a.init_debug();

	bool exiting = false;

	//#define VO

	#ifdef VO
	struct vo* vo = vo_open("out.mkv", size, size, "rgb32");
	#endif

	int mx = 0;
	int my = 0;
	bool paint = false;
	bool running = true;
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
					if(e.key.keysym.sym == SDLK_SPACE) running = !running;
					break;
				case SDL_QUIT:
					exiting = true;
					break;
			}
		}
		if(paint) {
			float m = 0.4f;
			int mmx = mx/2;
			int mmy = my/2;
			a.add_mass(mmx-1, mmy-1, m);
			a.add_mass(mmx, mmy-1, m);
			a.add_mass(mmx+1, mmy-1, m);
			a.add_mass(mmx-1, mmy, m);
			a.add_mass(mmx, mmy, m);
			a.add_mass(mmx+1, mmy, m);
			a.add_mass(mmx-1, mmy+1, m);
			a.add_mass(mmx, mmy+1, m);
			a.add_mass(mmx+1, mmy+1, m);
		}
		if(running) a.step();
		a.paint();

		#ifdef VO
		vo_frame(vo, a.bitmap);
		#endif

		SDL_UpdateTexture(t, NULL, a.bitmap, (1<<exp) * 4);
		SDL_SetRenderDrawColor(r, 255, 100, 0, 255);
		SDL_RenderClear(r);
		SDL_RenderCopy(r, t, NULL, NULL);
		SDL_RenderPresent(r);
	}

	#ifdef VO
	vo_close(vo);
	#endif

	return 0;
}
