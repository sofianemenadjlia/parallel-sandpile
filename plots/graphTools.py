# coding: 8859
import argparse
import pandas as pds
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import textwrap
from matplotlib.backends.backend_pdf import PdfPages


def openfile(path="./plots/data/perf_data.csv", sepa=";"):
    try:
        return pds.read_csv(path, sep=sepa)
    except FileNotFoundError:
        print("File not found: ", path, file=sys.stderr)
        sys.exit(1)

# Donne tous les champs de df qui ne sont pas listés


def complementaryCols(listeAttr, df):
    return [i for i in list(df.columns) if i not in listeAttr]


def constantCols(df):
    return df.columns[df.eq(df.iloc[0]).all()].tolist()


def shuffle(u, v, before=" ", equal="="):
    s = ""
    for i in zip(u, v):
        s += before + str(i[0]) + equal + str(i[1])
    return s


def factorizeLegendColRow(axisCol, args, axisAttr, df):
    if axisCol == None:
        return
    # list the attr that are functionally dependent on axisCol
    funcDeps = []
    for col in list(df.columns):
        if not col in axisAttr:
            projection = df[[axisCol, col]].groupby(axisCol).nunique()
            if projection[projection[col] > 1].empty:
                funcDeps += [col]
    if funcDeps != []:
        df["tmp"] = df[funcDeps].apply(
            lambda row: shuffle(funcDeps, row.values.tolist(), "  ", " = "), axis=1)
        for col in funcDeps:
            del df[col]
        df[axisCol] = df[axisCol].map(str) + " |" + df["tmp"]
        del df["tmp"]


def computeTitleAndLegend(args, df):
    axisAttr = [args.x, args.y, args.col, args.row, args.heaty]
    constantParameters = constantCols(df)
    if len(constantParameters) == 0:
        title = (u'Courbe de {y} en fonction de {x}').format(
            x=args.x, y=args.y)
    else:
        title = (u'{cons}').format(x=args.x, y=args.y,
                                   cons=textForConstantParameters(df, constantParameters))
        for col in constantParameters:
            if not col in axisAttr:
                del df[col]

    factorizeLegendColRow(args.col, args, axisAttr, df)
    factorizeLegendColRow(args.row, args, axisAttr, df)

    attr = complementaryCols(['time', 'ref'] + axisAttr, df)

    if attr != []:
        df['legend'] = df[attr].apply(
            lambda row: shuffle(attr, row.values.tolist()), axis=1)
    return title


# Cree les nom avec toutes les valeurs non utilisees pour le reste du graphique

def textForConstantParameters(df, constantParameters):
    string = ""
    # Donne les valeurs des constantes dans l'ordre des colonnes
    dataConst = [df[i].iloc[0] for i in constantParameters]
    for i in range(len(constantParameters)):
        if dataConst[i] != "none" and str(dataConst[i]) != "nan":
            string = string + \
                str(constantParameters[i]) + "=" + str(dataConst[i]) + " "
    return string


def computeSpeedUpAttr(df, args):  # Automatise la creation du speedup
    group = ['machine', 'size', 'kernel', 'iterations', 'arg']
    group = [
        attr for attr in group if attr not in args.delete and attr in list(df.columns)]
    if args.RefTimeVariants == [] and args.RefTimeTiling == [] :
        refDF = df[df.threads == 1].reset_index(drop=True)
    elif args.RefTimeTiling == []:
               refDF = df[df.variant.isin(args.RefTimeVariants)
                   & df.threads == 1].reset_index(drop=True)
    elif args.RefTimeVariants == []:
               refDF = df[df.tiling.isin(args.RefTimeTiling)
                   & df.threads == 1].reset_index(drop=True)    
    else:
        refDF = df[df.variant.isin(args.RefTimeVariants)
                   &  df.tiling.isin(args.RefTimeTiling)
                   & df.threads == 1].reset_index(drop=True)
    if refDF.empty:
        sys.exit("No row with OMP_NUM_THREADS=1 to compute speedUP")

    for i in complementaryCols(group + ['time'], refDF):
        del refDF[i]

    refDF = refDF.loc[refDF.groupby(
        group).time.idxmin()].reset_index(drop=True)
    refDF = refDF.rename(columns={'time': 'refTime'})

    df = df.merge(refDF, how='inner')

    df['speedup'] = df['refTime'] / df['time']
    del df['time']
    if args.noRefTime:
        del df['refTime']
    else:
        df['refTime'] = df['refTime'] // 1000
    return df


def heatFacet(*args, **kwargs):
    data = kwargs['data']
    data = data.pivot(index=args[1], columns=args[0], values=args[2])
    print("------------------------------------")
    print(data)
    print("------------------------------------")
    if "time (ms)" == args[2]:
        m = data.max().max()
        fmt = '.2f' if m < 10 else '.1f' if m < 100 else '.0f'
        g = sns.heatmap(data, cmap='rocket_r', annot=True,
                        fmt=fmt, annot_kws={"fontsize": 8})
    else:
        g = sns.heatmap(data, cmap='rocket', annot=True,
                        fmt='.2f', annot_kws={"fontsize": 8})
    g.invert_yaxis()
    plt.yticks(rotation=0)


def easyPlotDataFrame(df, args):
    title = computeTitleAndLegend(args, df)
    if args.y == "time":
        df['time'] = df['time'] / 1000
        df.rename(columns={'time': 'time (ms)'}, inplace=True)
        args.y = "time (ms)"

    legend = "legend" if "legend" in list(df.columns) else None

    if not args.no_sort:
        df = df.sort_values(by=args.y, ascending=False)

    if (args.plottype == 'lineplot'):
        g = sns.FacetGrid(df, row=args.row, col=args.col, hue=legend, sharex='col', sharey='row',
                          height=args.height, margin_titles=True, legend_out=not args.legendInside, aspect=args.aspect)
        g.map(sns.lineplot, args.x, args.y, err_style="bars", marker="o")
        g.add_legend()
    elif (args.plottype == 'heatmap'):
        df = df.groupby(complementaryCols([args.y], df), as_index=False).mean()
        g = sns.FacetGrid(df, row=args.row, col=args.col,
                          hue=legend, height=args.height, aspect=args.aspect)
        g = g.map_dataframe(heatFacet, args.x, args.heaty, args.y)
    else:
        g = sns.catplot(data=df, x=args.x, y=args.y, row=args.row, col=args.col, hue=legend,
                        kind=args.kind, sharex='col', sharey='row',
                        height=args.height, margin_titles=True, legend_out=not args.legendInside, aspect=args.aspect)

    if args.font_scale != 1.0:
        sns.set_context(font_scale=args.font_scale)
        plt.setp(g._legend.get_texts(), fontsize='11')  # for legend text
        plt.setp(g._legend.get_title(), fontsize='14')  # for legend title

    if args.yscale == "log2":
        plt.yscale("log", base=2)
    elif args.yscale != "linear":
        g.set(yscale=args.yscale)

    if args.xscale == "log2":
        plt.xscale("log", base=2)
    elif args.xscale != "linear":
        g.set(xscale=args.xscale)

    if not args.hideParameters:
        g.fig.suptitle(title, wrap=True)
        plt.subplots_adjust(top=args.adjustTop)

    else:
        print(title)

    g.tight_layout()
    return g


def parseArguments(argv):
    global parser, args
    parser = argparse.ArgumentParser(
        argv,
        description='''Process performance plots.
        The labels of the options are similar to those of
        easypap for all that relates to data selection,
        and to those of seaborn for all aspects of graphic
        layout https://seaborn.pydata.org/introduction.html''')

    all = ["size", "iterations", "kernel", "variant", "tiling", "threads",
           "nb_tiles", "schedule", "label", "machine", "tile_size", "tileh", "tilew", "arg", "places"]

    parser.add_argument("-x", "-heatx", choices=all +
                        ["custom"], default="threads")
    parser.add_argument("-heaty", choices=all+["custom"], default=None)
    parser.add_argument(
        "-y", choices=["time", "speedup", "throughput", "efficiency", "custom"], default="speedup")

    parser.add_argument('-rtv', '--RefTimeVariants',
                        action='store', nargs='+',
                        help="list of variants to take into account to compute the speedUP RefTimes",
                        default=[])
    
    parser.add_argument('-rtt', '--RefTimeTiling',
                        action='store', nargs='+',
                        help="list of tiling functions to take into account to compute the speedUP RefTimes",
                        default=[])

    parser.add_argument("-C", "--col", choices=all+["custom"], default=None)
    parser.add_argument("-R", "--row", choices=all+["custom"], default=None)

    parser.add_argument('-of', '--output',
                        action='store', nargs='?',
                        help='Filename to output the plot',
                        const='plot.pdf',
                        default='plot.pdf')

    parser.add_argument('-if', '--input',
                        action='store', nargs='?',
                        help="Data's filename",
                        const=os.getcwd() + "/plots/data/perf_data.csv",
                        default=os.getcwd() + "/plots/data/perf_data.csv")

    parser.add_argument('-k', '--kernel',
                        action='store', nargs='+',
                        help="list of kernels to plot",
                        default=[])

    parser.add_argument('-t', '--threads',
                        action='store', nargs='+', type=int,
                        help="list of numbers of threads to plot",
                        default=[])

    parser.add_argument('--delete',
                        action='store', nargs='+',
                        help="delete a column before proceeding data",
                        choices=all,
                        default=[]
                        )

    parser.add_argument('-v', '--variant',
                        action='store', nargs='+',
                        help="list of variants to plot",
                        default=[])

    parser.add_argument('-wt', '--tiling',
                        action='store', nargs='+',
                        help="list of tile functions to plot",
                        default=[])

    parser.add_argument('-th', '--tileh',
                        action='store', nargs='+', type=int,
                        help="list of tile heights to plot",
                        default=[])

    parser.add_argument('-tw', '--tilew',
                        action='store', nargs='+', type=int,
                        help="list of tile widths to plot",
                        default=[])

    parser.add_argument('-nt', '--nb_tiles',
                        action='store', nargs='+', type=int,
                        help="list of nb_tiles to plot",
                        default=[])


#    parser.add_argument('-ts', '--tile',
#                        action='store', nargs='*',type=int,
#                        help="print tile sizes rather than nb_tiles / list of tiles to plot",
#                        default=None)

    parser.add_argument('-m', '--machine',
                        action='store', nargs='+',
                        help="list of machines to plot",
                        default=[])

    parser.add_argument('-i', '--iterations',
                        action='store', nargs='+', type=int,
                        help="list of iterations to plot",
                        default=[])

    parser.add_argument('-sc', '--schedule',
                        action='store', nargs='+',
                        help="list of schedule policies to plot",
                        default=[])

    parser.add_argument('--places',
                        action='store', nargs='+',
                        help="list of schedule policies to plot",
                        default=[])

    parser.add_argument('--arg',
                        action='store', nargs='+',
                        help="list of arg value to plot",
                        default=[])

    parser.add_argument('-s', '--size',
                        action='store', nargs='+', type=int,
                        help="list of sizes to plot",
                        default=[])

    parser.add_argument('-lb', '--label',
                        action='store', nargs='+',
                        help="list of labels to plot",
                        default=[])

    parser.add_argument('--height',
                        action='store',
                        type=int,
                        help="to set the height of each subgraph",
                        default=4)

    parser.add_argument('--hideParameters',
                        action='store_true',
                        help="to hide constant parameters",
                        default=False)

    parser.add_argument('--legendInside',
                        action='store_true',
                        help="to print the legend inside the graph",
                        default=False)

    parser.add_argument("--no_sort",
                        action='store_true',
                        help="sort data following y",
                        default=False)

    parser.add_argument('--adjustTop',
                        action='store',
                        type=float,
                        help="to adjust the space for the suptitle",
                        default=1)

    parser.add_argument('--aspect',
                        action='store',
                        type=float,
                        help="to adjust the ratio length/height",
                        default=1.1)

    parser.add_argument('--font_scale',
                        action='store',
                        type=float,
                        help="to adjust the font of the title and the legend",
                        default=1.0)

    parser.add_argument('--noRefTime',
                        action='store_true',
                        help="do not print reftime in legend",
                        default=False)

    parser.add_argument('--yscale',
                        choices=["linear", "log", "log2", "symlog", "logit"],
                        action='store',
                        default="linear")

    parser.add_argument('--xscale',
                        choices=["linear", "log", "log2", "symlog", "logit"],
                        action='store',
                        default="linear")

    parser.add_argument('--plottype',
                        choices=['lineplot', 'catplot', 'heatmap'],
                        action='store',
                        default="lineplot")

    parser.add_argument('--kind',
                        choices=["strip", "swarm", "box", "violin",
                                 "boxen", "point", "bar", "count"],
                        help="kind of barplot (see sns catplot)",
                        action='store',
                        default="swarm")

    args = parser.parse_args()
    return args


def getDataFrame(args):
    # Lecture du fichier d'experiences:
    df = openfile(args.input, sepa=";")

    if args.kernel != []:
        df = df[df.kernel.isin(args.kernel)].reset_index(drop=True)

    if args.arg != []:
        df = df[df.arg.isin(args.arg)].reset_index(drop=True)

    if args.iterations != []:
        df = df[df.iterations.isin(args.iterations)].reset_index(drop=True)

    if args.size != []:
        df = df[df["size"].isin(args.size)].reset_index(drop=True)

    if args.machine != []:
        df = df[df.machine.isin(args.machine)].reset_index(drop=True)

    if args.delete != []:
        for attr in args.delete:
            del df[attr]

    if args.y == "speedup":
        df = computeSpeedUpAttr(df, args)

    if args.y == "efficiency":
        df = computeSpeedUpAttr(df, args)
        df['efficiency'] = df['speedup'] / df['threads']
        del df['speedup']

    if args.label != []:
        df = df[df.label.isin(args.label)].reset_index(drop=True)

    if args.schedule != []:
        df = df[df.schedule.isin(args.schedule)].reset_index(drop=True)

    if args.places != []:
        df = df[df.places.isin(args.places)].reset_index(drop=True)

    if args.threads != []:
        df = df[df.threads.isin(args.threads)].reset_index(drop=True)

    if args.variant != []:
        df = df[df.variant.isin(args.variant)].reset_index(drop=True)

    if args.tiling != []:
        df = df[df.tiling.isin(args.tiling)].reset_index(drop=True)

    if args.tileh != []:
        df = df[df.tileh.isin(args.tileh)].reset_index(drop=True)

    if args.tilew != []:
        df = df[df.tilew.isin(args.tilew)].reset_index(drop=True)

    if args.nb_tiles == []:
        if not ('tileh' in [args.col, args.row, args.x, args.heaty] or 'tilew' in [args.col, args.row, args.x,  args.heaty]):
            df['tile_size'] = df.tileh.map(
                str) + '$\\times$' + df.tilew.map(str)
            del df['tileh']
            del df['tilew']
    else:
        df['nb_tileh'] = df['size'] // df['tileh']
        df['nb_tilew'] = df['size'] // df['tilew']
        df = df[df.nb_tilew.isin(args.nb_tiles)].reset_index(drop=True)
        # df = df[df.nb_tileh.isin(args.nb_tiles)].reset_index(drop=True)
        df = df[df.nb_tileh == df.nb_tilew].reset_index(drop=True)
        df['nb_tiles'] = df['nb_tileh']
        del df['tileh']
        del df['tilew']
        del df['nb_tileh']
        del df['nb_tilew']

    if args.y == "throughput":
        args.y = 'throughput (MPixel / s)'
        df[args.y] = (df['size'] ** 2) * df['iterations'] / df['time']

    if df.empty:
        sys.exit("No data")

    # remove empty columns
    df = df.dropna(axis=1, how='all')
    return df.replace({None: "none"})


def savePlotAsPDF(fig):
    pp = PdfPages(args.output)
    plt.savefig(pp, format='pdf')
    pp.close()
