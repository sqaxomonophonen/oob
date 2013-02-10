#include <mach/mach_time.h>
#include "nanotime.h"
unsigned int mt_numer;
unsigned int mt_denom;
bool nanotime_initialized = false;

uint64_t nanotime_get() {
	if(!nanotime_initialized) {
		/* XXX MACH SPECIFIC */
		mach_timebase_info_data_t info;
		mach_timebase_info(&info);
		mt_numer = info.numer;
		mt_denom = info.denom;
		nanotime_initialized = true;
	}
	return ((mach_absolute_time() * mt_numer) / mt_denom);
}
