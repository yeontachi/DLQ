# Image to Column(im2col)

What is im2col
https://wikidocs.net/265961
row-major?

    // 블록/스레드에서 출력 위치(oh, ow) 담당
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    What this mean?

 // 경계 밖은 바로 반환
    if (oh >= OH || ow >= OW) return;
    어떤 경계?


https://teach-meaning.tistory.com/1294 thread block grid