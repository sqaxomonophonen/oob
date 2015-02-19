
#include <math.h>
#include <SDL.h>
#include <fftw3.h>
#include "nanotime.h"
#include <vector>

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
		uint64_t dt = nanotime_get() - t0;
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

struct solid {
	int width;
	int height;
	struct solid_cell* cells;

	float XXX_multiplier;

	struct vec2 position;
	float rotation;

	struct vec2 linear_velocity;
	float angular_velocity;
	float mass;
	struct vec2 force;
	struct vec2 center_of_mass;
	float torque;

	solid(int width, int height) : width(width), height(height) {
		cells = (struct solid_cell*) malloc(sizeof(struct solid_cell) * width * height);
		XXX_multiplier = 1.0f;
	}

	~solid() {
		free(cells);
	}

	void init_debug() {
		int r2 = width > height ? height*height/4 : width*width/4;
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				int border = 5;
				float m = (x < border || y < border || x > (width-border) || y > (height-border)) ? 5.2f : 0.0f;
				int dx = x-width/2;
				int dy = y-height/2;
				if((dx*dx+dy*dy) < r2) m = 3.1f;
				at(x,y).mass = m;
			}
		}
	}

	struct solid_cell& at(int x, int y) const {
		return cells[x + y * width];
	}

	void set(struct vec2& p, float r) {
		position = p;
		rotation = r;
	}

	void cell_update() { // XXX what if the solid is split into several? (flood fill)
		mass = 0.0f;
		center_of_mass = vec2();
		for(int y = 0; y < height; y++) {
			for(int x = 0; x < width; x++) {
				struct solid_cell& c = cells[x + y * width];
				mass += c.mass;
				struct vec2 p(x,y);
				p *= mass;
				center_of_mass += p;
			}
		}
		center_of_mass *= (1.0f / mass);
	}

	void step() {
		vec2 fm = force / mass;
		linear_velocity += fm;
		force = vec2(0,0);
		angular_velocity += torque / mass;
		torque = 0.0f;
		position += linear_velocity;
		rotation += angular_velocity;
	}

	void apply_force(struct vec2& applied_force, struct vec2& to_point) {
		force += applied_force;
		torque += (to_point - center_of_mass).cross(force); // XXX m_sweep?
	}
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
			data[i][1] = data[i][1] * kernel[i][0] + data[i][0] * kernel[i][1];
			data[i][0] = new_re;
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
				if(d < 34000 && d > 30000) {
					at(x,y).mass = 0.47;
					at(x,y).vx = dy * 0.0007f + dx * 0.001f;
					at(x,y).vy = -dx * 0.0007f + dy * 0.001f;
				}
				if(d <= 1) {
					//at(x,y).mass = 2.1f;
				}
			}
		}
		// XXX TODO clean up!
		struct solid* solid = new struct solid(280, 120);
		solid->position = vec2(256, 256);
		solid->rotation = 0.0f;
		solid->init_debug();
		solids.push_back(solid);
		 
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

	void rasterize_solid_trapezoid(struct solid& s, float y1f, float y2f, float x1f, float x2f, float x3f, float x4f, const struct vec2& t0, const struct vec2& dt) {
		// y
		int y1 = y1f;
		int y2 = y2f;
		int dy = y2 - y1;

		// left
		int x_l = x1f;
		int dx_l = abs(((int)x3f) - x_l);
		int sx_l = x3f>x1f ? 1 : -1;
		int err_l = dx_l - dy;

		// right
		int x_r = x2f;
		int dx_r = abs(((int)x4f) - x_r);
		int sx_r = x4f>x2f ? 1 : -1;
		int err_r = dx_r - dy;


		const float M = 65536.0f;
		float u0 = t0.u;
		float v0 = t0.v;
		float du = dt.u;
		float dv = dt.v;
		int u = u0 * M;
		int v = v0 * M;
		int dxu = du * M;
		int dxv = dv * M;
		int dyu = -dv * M;
		int dyv = du * M;
		int tsxu_l = x3f>x1f ? dxu : -dxu;
		int tsxv_l = x3f>x1f ? dxv : -dxv;

		int OOB = 0;
		int TOOB = 0;
		for(int y = y1; y < y2; y++) {
			int save_u = u;
			int save_v = v;
			for(int x = x_l; x < x_r; x++) {
				if(x < 0 || y < 0 || x >= length() || y >= length()) {
					OOB++; // XXX convert to exit(1) at some point, then remove completely when it's safe
				} else {
					int nu = u >> 16;
					int nv = v >> 16;
					if(nu < 0 || nv < 0 || nu >= s.width || nv >= s.height) {
						TOOB++; // XXX convert to exit(1) at some point, then remove completely when it's safe
					} else {
						//at(x,y).mass = s.at(nu, nv).mass;
						at(x,y).mass += s.at(nu, nv).mass * s.XXX_multiplier; // XXX
						atz(x,y).XXX_residual = s.at(nu, nv).mass;
					}
				}
				u += dxu;
				v += dxv;
			}
			u = save_u;
			v = save_v;
			while(1) {
				int e2 = err_l << 1;
				if(e2 > -dy) {
					err_l -= dy;
					x_l += sx_l;
					u += tsxu_l;
					v += tsxv_l;
				}
				if(e2 < dx_l) {
					err_l += dx_l;
					break;
				}
			}
			while(1) {
				int e2 = err_r << 1;
				if(e2 > -dy) {
					err_r -= dy;
					x_r += sx_r;
				}
				if(e2 < dx_r) {
					err_r += dx_r;
					break;
				}
			}
			u += dyu;
			v += dyv;
		}
		if(OOB > 0) {
			//printf("%d pixels out of bounds\n", OOB);
		}
		if(TOOB > 0) {
			//printf("%d texels out of bounds\n", TOOB);
		}
	}

	void rasterize_solid_trapezoid_clip(struct solid& s, float y1f, float y2f, float x1f, float x2f, float x3f, float x4f, struct vec2 t0, struct vec2 dt) {
		// clip y
		if(y2f <= y1f) return; // XXX check probably not required
		if(y2f < 1.0f) return;
		if(y1f >= length()) return;
		if(y1f < 0.0f) {
			float i = -y1f / (y2f - y1f); // y1f + (y2f - y1f) * i = 0
			float old_x1f = x1f;
			x1f = lerp<float>(x1f, x3f, i);
			x2f = lerp<float>(x2f, x4f, i);
			t0 += dt * (x1f - old_x1f);
			t0 += dt.normal() * -y1f;
			y1f = 0.0f;
		}

		if(y2f > length()) {
			float i = (length() - y1f) / (y2f - y1f); // y1f + (y2f - y1f) * i = length()
			x3f = lerp<float>(x1f, x3f, i);
			x4f = lerp<float>(x2f, x4f, i);
			y2f = length();
		}

		// clip left
		if(x1f >= length() && x3f >= length()) {
			return;
		} else if(x1f < 0.0f && x3f < 0.0f) {
			t0 += dt * -x1f;
			x1f = 0.0f;
			x3f = 0.0f;
		} else if(x1f < 0.0f || x3f < 0.0f) {
			// partly outside on the left; clip recursively
			float i = -x1f / (x3f-x1f); // x1f + (x3f-x1f) * i = 0
			float mx = lerp<float>(x2f, x4f, i);
			float my = lerp<float>(y1f, y2f, i);
			if(x1f < 0.0f) {
				t0 += dt * -x1f;
				rasterize_solid_trapezoid_clip(s, y1f, my, 0.0f, x2f, 0.0f, mx, t0, dt);
				t0 += dt.normal() * (my - y1f);
				rasterize_solid_trapezoid_clip(s, my, y2f, 0.0f, mx, x3f, x4f, t0, dt);
			} else {
				rasterize_solid_trapezoid_clip(s, y1f, my, x1f, x2f, 0.0f, mx, t0, dt);
				t0 += dt * -x1f + dt.normal() * (my - y1f);
				rasterize_solid_trapezoid_clip(s, my, y2f, 0.0f, mx, 0.0f, x4f, t0, dt);
			}
			return;
		}

		// clip right
		if(x2f < 1.0f && x4f < 1.0f) {
			return;
		} else if(x2f > length() && x4f > length()) {
			x2f = length();
			x4f = length();
		} else if(x2f > length() || x4f > length()) {
			// partly outside on the right; clip recursively
			float i = (length() - x2f) / (x4f-x2f); // x2f + (x4f-x2f) * i = length()
			float mx = lerp<float>(x1f, x3f, i);
			float my = lerp<float>(y1f, y2f, i);
			if(x2f > length()) {
				rasterize_solid_trapezoid_clip(s, y1f, my, x1f, length(), mx, length(), t0, dt);
				t0 += dt * (mx - x1f) + dt.normal() * (my - y1f);
				rasterize_solid_trapezoid_clip(s, my, y2f, mx, length(), x3f, x4f, t0, dt);
			} else {
				rasterize_solid_trapezoid_clip(s, y1f, my, x1f, x2f, mx, length(), t0, dt);
				t0 += dt * (mx - x1f) + dt.normal() * (my - y1f);
				rasterize_solid_trapezoid_clip(s, my, y2f, mx, length(), x3f, length(), t0, dt);
			}
			return;
			
		}

		rasterize_solid_trapezoid(s, y1f, y2f, x1f, x2f, x3f, x4f, t0, dt);
	}

	void rasterize_solid(struct solid& s) {
		scope_timer tt("rasterize solid");

		// rotation vector
		vec2 r(cosf(s.rotation), sinf(s.rotation));

		// origin
		vec2 p0 = s.position;

		// width corner displacement
		vec2 dw = r * s.width;

		// width corner position
		vec2 pw = p0 + dw;

		// height corner displacement
		vec2 dh = r.normal() * s.height;

		// height corner position
		vec2 ph = p0 + dh;

		// opposite corner position
		vec2 p1 = p0 + dw + dh;

		// texture corners
		struct vec2 t0 = vec2(0.5f, 0.5f);
		struct vec2 tw = vec2(s.width-1, 0.5f);
		struct vec2 th = vec2(0.5f, s.height-1);
		struct vec2 t1 = tw + th;

		// texture gradient
		struct vec2 dt = vec2(r.u, -r.v);

		// warp rotation into [0;2pi[ and fint quadrant
		float a = s.rotation;
		while(a < 0.0f) a += PI2;
		while(a >= PI2) a -= PI2;
		int q = (a / HALFPI);
		switch(q) {
		case 0:
			if(dw.v < dh.v) {
				float i = dw.v / dh.v;
				float m0 = lerp<float>(p0.u, ph.u, i);
				float m1 = lerp<float>(pw.u, p1.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, p0.v, pw.v, p0.u, p0.u, m0, pw.u, t0, dt);
				rasterize_solid_trapezoid_clip(s, pw.v, ph.v, m0, pw.u, ph.u, m1, th*i, dt);
				rasterize_solid_trapezoid_clip(s, ph.v, p1.v, ph.u, m1, p1.u, p1.u, th, dt);
			} else {
				float i = dh.v / dw.v;
				float m0 = lerp<float>(p0.u, pw.u, i);
				float m1 = lerp<float>(ph.u, p1.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, p0.v, ph.v, p0.u, p0.u, ph.u, m0, t0, dt);
				rasterize_solid_trapezoid_clip(s, ph.v, pw.v, ph.u, m0, m1, pw.u, th, dt);
				rasterize_solid_trapezoid_clip(s, pw.v, p1.v, m1, pw.u, p1.u, p1.u, tw*(1-i)+th, dt);
			}
			break;
		case 1:
			if(p0.v < p1.v) {
				float i = (p0.v-ph.v)/(p1.v-ph.v);
				float m0 = lerp<float>(ph.u, p1.u, i);
				float m1 = lerp<float>(p0.u, pw.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, ph.v, p0.v, ph.u, ph.u, m0, p0.u, th, dt);
				rasterize_solid_trapezoid_clip(s, p0.v, p1.v, m0, p0.u, p1.u, m1, tw*i+th, dt);
				rasterize_solid_trapezoid_clip(s, p1.v, pw.v, p1.u, m1, pw.u, pw.u, t1, dt);
			} else {
				float i = (ph.v-p1.v)/(ph.v-p0.v);
				float m0 = lerp<float>(ph.u, p0.u, i);
				float m1 = lerp<float>(p1.u, pw.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, ph.v, p1.v, ph.u, ph.u, p1.u, m0, th, dt);
				rasterize_solid_trapezoid_clip(s, p1.v, p0.v, p1.u, m0, m1, p0.u, t1, dt);
				rasterize_solid_trapezoid_clip(s, p0.v, pw.v, m1, p0.u, pw.u, pw.u, tw + th*i, dt);
			}
			break;
		case 2:
			if(dh.v < dw.v) {
				float i = (ph.v-p1.v)/(pw.v-p1.v);
				float m0 = lerp<float>(p1.u, pw.u, i);
				float m1 = lerp<float>(ph.u, p0.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, p1.v, ph.v, p1.u, p1.u, m0, ph.u, t1, dt);
				rasterize_solid_trapezoid_clip(s, ph.v, pw.v, m0, ph.u, pw.u, m1, tw+th*(1-i), dt);
				rasterize_solid_trapezoid_clip(s, pw.v, p0.v, pw.u, m1, p0.u, p0.u, tw, dt);
			} else {
				float i = (pw.v-p1.v)/(ph.v-p1.v);
				float m0 = lerp<float>(p1.u, ph.u, i);
				float m1 = lerp<float>(pw.u, p0.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, p1.v, pw.v, p1.u, p1.u, pw.u, m0, t1, dt);
				rasterize_solid_trapezoid_clip(s, pw.v, ph.v, pw.u, m0, m1, ph.u, tw, dt);
				rasterize_solid_trapezoid_clip(s, ph.v, p0.v, m1, ph.u, p0.u, p0.u, tw*i, dt);
			}
			break;
		case 3:
			if(p1.v < p0.v) {
				float i = (p1.v-pw.v)/(p0.v-pw.v);
				float m0 = lerp<float>(pw.u, p0.u, i);
				float m1 = lerp<float>(p1.u, ph.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, pw.v, p1.v, pw.u, pw.u, m0, p1.u, tw, dt);
				rasterize_solid_trapezoid_clip(s, p1.v, p0.v, m0, p1.u, p0.u, m1, tw*(1-i), dt);
				rasterize_solid_trapezoid_clip(s, p0.v, ph.v, p0.u, m1, ph.u, ph.u, t0, dt);
			} else {
				float i = (pw.v-p0.v)/(pw.v-p1.v);
				float m0 = lerp<float>(pw.u, p1.u, i);
				float m1 = lerp<float>(p0.u, ph.u, 1.0f - i);
				rasterize_solid_trapezoid_clip(s, pw.v, p0.v, pw.u, pw.u, p0.u, m0, tw, dt);
				rasterize_solid_trapezoid_clip(s, p0.v, p1.v, p0.u, m0, m1, p1.u, t0, dt);
				rasterize_solid_trapezoid_clip(s, p1.v, ph.v, m1, p1.u, ph.u, ph.u, th*(1-i), dt);
			}
			break;
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

		// rasterize and step solids
		for(std::vector<struct solid*>::iterator s = solids.begin(); s != solids.end(); s++) {
			(*s)->XXX_multiplier = 1.0f;
			rasterize_solid(**s);
			//s->step();
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

		// erase-rasterize and step solids
		for(std::vector<struct solid*>::iterator s = solids.begin(); s != solids.end(); s++) {
			(*s)->XXX_multiplier = -1.0f;
			rasterize_solid(**s);
			(*s)->rotation += 0.01f;
			//s->step();
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

	void paint(SDL_Surface* screen) {
		SDL_LockSurface(screen);
		char* pixels = (char*) screen->pixels;
		for(int i = 0; i < area(); i++) {
			int m1 = cells[i].d[d].mass * 500.0f;
			int m2 = cells[i].d[d].mass * 150.0f;
			int m3 = (cells[i].d[d].mass * cells[i].d[d].mass + cells[i].d[d].XXX_residual) * 100.0f;
			pixels[i*screen->format->BytesPerPixel+1] = m1 > 255 ? 255 : m1;
			pixels[i*screen->format->BytesPerPixel+2] = m2 > 255 ? 255 : m2;
			pixels[i*screen->format->BytesPerPixel+3] = cells[i].d[d].mass != 0 ? (m3 < 30 ? 30 : (m3 > 255 ? 255 : m3)) : 0;
			//pixels[i*screen->format->BytesPerPixel+3] = m3 > 70 ? 70 : m3;
			//pixels[i*screen->format->BytesPerPixel+3] = cells[i].d[d].mass != 0 ? 30 : 0;
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
		if(running) a.step();
		a.paint(screen);
		SDL_Flip(screen);
	}

	return 0;
}

