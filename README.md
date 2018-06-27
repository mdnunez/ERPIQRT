<img src="./extra/Differentielle_Psychologie.png" height="128"> <img src="./extra/small_cidlab_logo.png" height="128">

### Citation

Schubert, A. L., Nunez, M. D., Hagemann, D., & Vandekerckhove, J. (2018). Individual differences in cognitive abilities are predicted by cortical processing speed: A model-based cognitive neuroscience account. Manuscript in preparation.

# ERPIQRT
#### (Repository version 0.3.1)

Individual differences in cognitive abilities are predicted by cortical processing speed: A model-based cognitive neuroscience account (ERPIQRT).

**Authors: Anna-Lena Schubert and Dirk Hagemann from Heidelberg University, Germany and Michael D. Nunez and Joachim Vandekerckhove from the University of California, Irvine, USA**

### Research Questions

A greater speed of information-processing may facilitate evidence acquisition during decision making and memory updating and give rise to advantages in general cognitive abilities. In this study we explore this hypothesis by using a hierarchical Bayesian cognitive modeling approach to investigate if individual differences in the velocity of evidence accumulation (a drift-diffusion model parameter that explains some variance in reaction times; RT) mediates the relationship between neural processing (estimated by event-related potentials; ERPs) and general cognitive abilities (measured by intelligence tests; IQ)

### Hypothesis

Individual differences in cognitive abilities, a latent variable related to IQ scores, are explained by individual differences in ERP latencies. This relationship is at least partially reflected in individual differences in evidence accumulation (estimated by drift-diffusion model parameter fits).

### Prerequisites

[MATLAB](https://www.mathworks.com/)

[MCMC Sampling Program: JAGS](http://mcmc-jags.sourceforge.net/)

[Program: JAGS Wiener module](https://sourceforge.net/projects/jags-wiener/)

[Scientific Python libraries](https://www.continuum.io/downloads)

[Python Repository: pyjags](https://github.com/tmiasko/pyjags)

### Downloading

The repository can be cloned with `git clone https://github.com/mdnunez/ERPIQRT.git`

The repository can also be may download via the _Download zip_ button above.

### Installation

After downloading/unzipping the repository, users will need to add these functions to the MATLAB path. In MATLAB, add the repository to the PATH with

```matlab
%Set 'artloc' to full directory path
emloc = 'C:\Users\MATLAB\ERPIQRT';
addpath(genpath(emloc));
```

### License

ERPIQRT is licensed under the GNU General Public License v3.0 and written by Anna-Lena Schubert, Michael D. Nunez, Dirk Hagemann, and Joachim Vandekerckhove.

### Further Reading

Schubert, A. L., Hagemann, D., Voss, A., Schankin, A., & Bergmann, K. (2015). [Decomposing the relationship between mental speed and mental abilities.](http://www.psychologie.uni-heidelberg.de/ae/meth/team/voss/paper/schubert_et_al_2015.pdf) Intelligence, 51, 28-46.

Schubert, A. L., Hagemann, D., & Frischkorn, G. T. (2017). [Is general intelligence little more than the speed of higher-order processing?.](http://psycnet.apa.org/record/2017-30267-001) Journal of Experimental Psychology: General, 146(10), 1498.


