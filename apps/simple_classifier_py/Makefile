
GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
NOCOLOR = '\033[0m'

# name of the directory and the base name of the main python file (minus the .py extension)
APP_NAME = simple_classifier_py

# filenames for the graph files that we'll copy to this directory.
CLASSIFIER_RELATIVE_DIR = ../../apps/simple_classifier_py
TEST_IMAGE_RELATIVE_DIR = ../../data/images
LABELS_RELATIVE_DIR = ../../data/ilsvrc12
MODEL_RELATIVE_DIR = ../../networks/googlenet_v1

TEST_IMAGE_FILENAME = nps_electric_guitar.png
LABELS_FILENAME = synset_labels.txt

MODEL_FILE_NAME_BASE = googlenet-v1


.PHONY: all
all: deps data


.PHONY: deps
deps: default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making dependencies..."$(NOCOLOR)


.PHONY: data
data:
	@echo $(YELLOW)'\n'${APP_NAME}": Making the required data..."$(NOCOLOR)
	@(cd ../../data/ilsvrc12; make all;)


.PHONY: compile_default_model
compile_default_model: 
	@echo $(YELLOW)'\n'${APP_NAME}": Compiling the model to IR..."$(NOCOLOR)
	@if [ -e ${MODEL_FILE_NAME_BASE}.bin ] && [ -e ${MODEL_FILE_NAME_BASE}.xml ]; \
	then \
		echo "- Googlenet IR files found. No need to compile."; \
	else \
		echo "- Making googlenet IRs in googlenet project."; \
		(cd ${MODEL_RELATIVE_DIR}; make compile_model;); \
		echo "- Copying IRs to project folder..."; \
		(cd ${MODEL_RELATIVE_DIR}; cp ${MODEL_FILE_NAME_BASE}.bin ${CLASSIFIER_RELATIVE_DIR}; cp ${MODEL_FILE_NAME_BASE}.xml ${CLASSIFIER_RELATIVE_DIR};); \
	fi

	
.PHONY: default_model
default_model: compile_default_model
	@echo $(YELLOW)'\n'${APP_NAME}": Making default models..."$(NOCOLOR)
	
	
.PHONY: run
run: run_py


.PHONY: run_py
run_py: deps data
	@echo $(YELLOW)'\n'${APP_NAME}": Running the sample..."$(NOCOLOR)
	@if [ -z "$(INTEL_OPENVINO_DIR)" ] ; \
	then \
		echo "Please initiate the Intel OpenVINO environment by going to the installation directory for openvino and running the setupvars.sh file in the bin folder." ; \
		exit 1 ; \
	else \
		echo "Intel OpenVINO environment is already set!" ; \
	fi
	python3 run.py --ir=${MODEL_FILE_NAME_BASE}.xml --image=${TEST_IMAGE_RELATIVE_DIR}/${TEST_IMAGE_FILENAME} --labels=${LABELS_RELATIVE_DIR}/${LABELS_FILENAME} 


.PHONY: install-reqs
install-reqs: 
	@echo $(YELLOW)"\n"$(APP_NAME)": Checking installation requirements..."$(NOCOLOR)
	@echo "No requirements needed."
	

.PHONY: uninstall-reqs
uninstall-reqs: 
	@echo $(YELLOW)'\n'${APP_NAME}": Uninstalling requirements..."$(NOCOLOR)
	@echo "Nothing to uninstall."
	
	
.PHONY: help
help:
	@echo "\nPossible make targets: ";
	@echo $(YELLOW)"  make run or run_py"$(NOCOLOR)"- runs the application";
	@echo $(YELLOW)"  make help "$(NOCOLOR)"- shows this message";
	@echo $(YELLOW)"  make all "$(NOCOLOR)"- makes everything needed to run but doesn't run";
	@echo $(YELLOW)"  make data "$(NOCOLOR)"- downloads data as needed";
	@echo $(YELLOW)"  make deps "$(NOCOLOR)"- makes/prepares dependencies";
	@echo $(YELLOW)"  make install-reqs "$(NOCOLOR)"- Installs requirements needed to run this sample on your system.";
	@echo $(YELLOW)"  make uninstall-reqs "$(NOCOLOR)"- Uninstalls requirements that were installed by the sample program.";
	@echo $(YELLOW)"  make default_model "$(NOCOLOR)"- compiles a default model to use when running";
	@echo $(YELLOW)"  make clean "$(NOCOLOR)"- removes all created content";
	@echo ""


clean: 
	@echo $(YELLOW)'\n'${APP_NAME}": Cleaning up files..."$(NOCOLOR)
	rm -f ${MODEL_FILE_NAME_BASE}.xml
	rm -f ${MODEL_FILE_NAME_BASE}.bin
