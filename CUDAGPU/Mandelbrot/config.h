#define WIDTH 2048
#define HEIGHT 2048

#define MAX_ITERATION 5000

#define THREADS 1024
#define REPEAT 10

#define IMAGE "./%s_%02d.png"
#define REPORT "./report.txt"

#define BLOCK_SIZE THREADS
#define LENGTH WIDTH * HEIGHT
#define GRID_SIZE LENGTH / BLOCK_SIZE
