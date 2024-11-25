from matplotlib import pyplot as plt
import matplotlib as mpl

plt.style.use('seaborn-v0_8-paper')
# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
# plt.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
mpl.rcParams['xtick.major.pad'] = 0.
mpl.rcParams['ytick.major.pad'] = 0.
mpl.rcParams['axes.labelpad'] = 0