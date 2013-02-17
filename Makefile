# Makefile

CC = clang++
CCFLAGS = -m64 -O3 -Wall -I/Library/Frameworks/SDL.framework/Headers -I${HOME}/usr/include
#LINK = -framework Cocoa -framework SDL -L${HOME}/usr/lib -lfftw3f
LINK = -framework Cocoa -framework SDL -L${HOME}/usr/lib -lfftw3f
O=.o
OWNOBJS = main.o nanotime.o
ALLOBJS = $(OWNOBJS) SDLMain.o
DEPS = nanotime.h


all: main

%.o: %.cc $(DEPS)
	$(CC) -c -o $@ $< $(CCFLAGS)

SDLMain.o: SDLMain.m
	$(CC) $(CCFLAGS) -c SDLMain.m

main: $(ALLOBJS) Makefile
	$(CC) $(CCFLAGS) -o main \
		$(ALLOBJS) \
		$(LINK)

clean:
	rm -f $(OWNOBJS) main

cleanall:
	rm -f $(ALLOBJS) main
