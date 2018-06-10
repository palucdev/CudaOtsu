SOURCE_DIR = src
BUILD_DIR = build
EXEC_FILE = CudaOtsu
DEFAULT_VALUE_FLAG = -1

SOURCE_FILES := $(shell find $(SOURCEDIR) -name '*.cpp' -o -name '*.cu')

build:
	mkdir -p ${BUILD_DIR}
	nvcc -x cu ${SOURCE_FILES} --std=c++11 -lineinfo -o ${BUILD_DIR}/${EXEC_FILE}

run:
	./${BUILD_DIR}/${EXEC_FILE} $(file) $(threads) $(blocks) -d $(device_id)

run_default:
	./${BUILD_DIR}/${EXEC_FILE} $(file) ${DEFAULT_VALUE_FLAG} ${DEFAULT_VALUE_FLAG} -d 0

run_histogram:
	./${BUILD_DIR}/${EXEC_FILE} $(file) $(threads) $(blocks) -d $(device_id) -h

clean:
	rm -rf ${BUILD_DIR}
