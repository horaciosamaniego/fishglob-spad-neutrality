.PHONY: all setup quick shares fits test power subsets gof figure regions profile clean
# `regions` also refreshes the copy of Figure 2 that the GitHub Pages site serves;
# docs/ cannot reach ../figures/ once published, so it keeps its own copy.

all: test power subsets gof figure regions

setup:
	pip install -r requirements.txt

quick:
	python src/run_analysis.py --quick

shares:
	python src/run_analysis.py --stage shares

fits:
	python src/run_analysis.py --stage fits | tee output/fits.txt

test:
	python src/run_analysis.py --stage boot | tee output/test.txt

power:
	python src/run_analysis.py --stage power | tee output/power.txt

subsets:
	python src/run_analysis.py --stage subsets | tee output/subsets.txt

gof:
	python src/run_analysis.py --stage gof | tee output/gof.txt

figure:
	python src/run_analysis.py --stage figure

profile:
	python src/profile_surveys.py | tee output/survey_profile.txt

regions:
	python src/run_regions.py | tee output/regions.txt
	python src/make_region_figure.py
	cp figures/nu_test_across_regions.png docs/nu_test_across_regions.png

clean:
	rm -f output/*.txt output/boot_h0.pkl
	find . -name __pycache__ -type d -exec rm -rf {} +
