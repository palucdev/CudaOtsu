SOURCE_DIR = src
BUILD_DIR = build
EXEC_FILE = CudaOtsu

SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp' -o -name '*.cu')

build:
	mkdir -p ${BUILD_DIR}
	nvcc -x cu ${SOURCE_FILES} --std=c++11 -lineinfo -o ${BUILD_DIR}/${EXEC_FILE}

run:
	./${BUILD_DIR}/${EXEC_FILE} $(file) $(threads) $(blocks) $(histograms)

clean:
	rm -rf ${BUILD_DIR}
