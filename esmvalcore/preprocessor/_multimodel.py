"""Functions to compute multi-cube statistics."""

import logging
from datetime import datetime
from functools import reduce

import cf_units
import iris
import numpy as np
from iris.experimental.equalise_cubes import equalise_attributes

logger = logging.getLogger(__name__)


def _get_consistent_time_unit(cubes):
    """Return cubes' time unit if consistent, standard calendar otherwise."""
    t_units = [cube.coord('time').units for cube in cubes]
    if len(set(t_units)) == 1:
        return t_units[0]
    return cf_units.Unit("days since 1850-01-01", calendar="standard")


def _unify_time_coordinates(cubes):
    """Make sure all cubes' share the same time coordinate.

    This function extracts the date information from the cube and
    reconstructs the time coordinate, resetting the actual dates to the
    15th of the month or 1st of july for yearly data (consistent with
    `regrid_time`), so that there are no mismatches in the time arrays.

    If cubes have different time units, it will use reset the calendar to
    a default gregorian calendar with unit "days since 1850-01-01".

    Might not work for (sub)daily data, because different calendars may have
    different number of days in the year.
    """
    t_unit = _get_consistent_time_unit(cubes)

    for cube in cubes:
        # Extract date info from cube
        coord = cube.coord('time')
        years = [p.year for p in coord.units.num2date(coord.points)]
        months = [p.month for p in coord.units.num2date(coord.points)]
        days = [p.day for p in coord.units.num2date(coord.points)]

        # Reconstruct default calendar
        if 0 not in np.diff(years):
            # yearly data
            dates = [datetime(year, 7, 1, 0, 0, 0) for year in years]
        elif 0 not in np.diff(months):
            # monthly data
            dates = [
                datetime(year, month, 15, 0, 0, 0)
                for year, month in zip(years, months)
            ]
        elif 0 not in np.diff(days):
            # daily data
            dates = [
                datetime(year, month, day, 0, 0, 0)
                for year, month, day in zip(years, months, days)
            ]
            if coord.units != t_unit:
                logger.warning(
                    "Multimodel encountered (sub)daily data and inconsistent "
                    "time units or calendars. Attempting to continue, but "
                    "might produce unexpected results.")
        else:
            raise ValueError(
                "Multimodel statistics preprocessor currently does not "
                "support sub-daily data.")

        # Update the cubes' time coordinate (both point values and the units!)
        cube.coord('time').points = t_unit.date2num(dates)
        cube.coord('time').units = t_unit
        cube.coord('time').bounds = None
        cube.coord('time').guess_bounds()


def _interpolate(cubes, func):
    """Expand or subset cubes so they share a common time span."""
    _unify_time_coordinates(cubes)
    time_spans = [cube.coord('time').points for cube in cubes]
    new_times = reduce(func, time_spans)
    # new_times = cubes[0].coord('time').units.num2date(new_times)
    sample_points = [('time', new_times)]
    scheme = iris.analysis.Nearest(extrapolation_mode='nan')
    return [cube.interpolate(sample_points, scheme) for cube in cubes]


def _extend(cubes):
    return _interpolate(cubes, np.intersect1d)


def _subset(cubes):
    """Only keep the times that are present in all cubes."""
    return _interpolate(cubes, np.union1d)


def _align(cubes, span):
    """Expand or subset cubes so they share a common time span."""

    if span == 'overlap':
        new_cubes = _subset(cubes)
    elif span == 'full':
        new_cubes = _extend(cubes)
    else:
        raise ValueError("Unknown value for span. Expected 'full' or 'overlap'"
                         "got {}".format(span))

    for cube in new_cubes:
        cube.coord('time').guess_bounds()
    return new_cubes


def _combine(cubes, dim='new_dim'):
    """Merge iris cubes into a single big cube with new dimension."""
    equalise_attributes(cubes)

    for i, cube in enumerate(cubes):
        concat_dim = iris.coords.AuxCoord(i, var_name=dim)
        cube.add_aux_coord(concat_dim)

    cubes = iris.cube.CubeList(cubes)
    return cubes.merge_cube()


def _compute(cube, statistic, dim='new_dim'):
    """Compute statistic."""
    operators = vars(iris.analysis)

    try:
        operator = operators[statistic.upper()]
    except KeyError as err:
        raise ValueError(
            f'Statistic `{statistic}` not supported in',
            '`ensemble_statistics`. Choose supported operator from',
            '`iris.analysis package`.') from err

    # This will always return a masked array
    return cube.collapsed('concat_dim', operator)


def _multicube_statistics(cubes, statistics, span):
    """Compute multi-cube statistics.

    Cubes are merged and subsequently collapsed along a new auxiliary
    coordinate. Inconsistent attributes will be removed.

    This function deals with non-homogeneous cubes by taking the time union
    computed across a common overlap in time (set span: overlap) or across the
    full length in time of each model (set span: full). Apart from the time
    coordinate, cubes must have consistent shapes.

    This method uses iris' built in functions, exposing the operators in
    iris.analysis and supporting lazy evaluation.

    Note: some of the operators in iris.analysis require additional
    arguments, such as percentiles or weights. These operators are
    currently not supported.

    Parameters
    ----------
    cubes: list
        list of cubes to be used in multimodel stat computation;
    statistics: list
        statistical measures to be computed. Choose from the
        operators listed in the iris.analysis package.
    span: str
        'overlap' or 'full'. If 'overlap', statitsticss are computed on common
        time span; if 'full', statistics are computed on full time spans,
        ignoring missing data.

    Returns
    -------
    dict
        dictionary of statistics cubes with statistics' names as keys.
    """
    aligned_cubes = _align(cubes, span=span)
    big_cube = _combine(aligned_cubes)
    statistics_cubes = {}
    for statistic in statistics:
        statistics_cubes[statistic] = _compute(big_cube, statistic)

    return statistics_cubes


def _multiproduct_statistics(products, statistics, output_products, span=None):
    """Compute multi-cube statistics on ESMValCore products.

    Extract cubes from products, calculate multicube statistics and
    assign the resulting output cubes to the output_products.
    """
    cubes = [cube for product in products for cube in product.cubes]
    statistics_cubes = _multicube_statistics(cubes=cubes,
                                             statistics=statistics,
                                             span=span)
    statistics_products = set()
    for statistic, cube in statistics_cubes.items():
        statistics_product = output_products[statistic]
        statistics_product.cubes = [cube]

        for product in products:
            statistics_product.wasderivedfrom(product)

        logger.info("Generated %s", statistics_product)
        statistics_products.add(statistics_product)

    return statistics_products


def multi_model_statistics(products, span, statistics, output_products=None):
    """Compute multi-model statistics.

    This function computes multi-model statistics on cubes or products.
    Products (or: preprocessorfiles) are used internally by ESMValCore to store
    workflow and provenance information, and this option should typically be
    ignored.

    This function was designed to work on (max) four-dimensional data: time,
    vertical axis, two horizontal axes. Apart from the time coordinate, cubes
    must have consistent shapes. There are two options to combine time
    coordinates of different lengths, see the `span` argument.

    Parameters
    ----------
    products: list
        Cubes (or products) over which the statistics will be computed.
    statistics: list
        Statistical metrics to be computed. Available options: mean, median,
        max, min, std, or pXX.YY (for percentile XX.YY; decimal part optional).
    span: str
        Overlap or full; if overlap, statitstics are computed on common time-
        span; if full, statistics are computed on full time spans, ignoring
        missing data.
    output_products: dict
        For internal use only. A dict with statistics names as keys and
        preprocessorfiles as values. If products are passed as input, the
        statistics cubes will be assigned to these output products.

    Returns
    -------
    dict
        A dictionary of statistics cubes with statistics' names as keys. (If
        input type is products, then it will return a set of output_products.)

    Raises
    ------
    ValueError
        If span is neither overlap nor full, or if input type is neither cubes
        nor products.
    """
    if all(isinstance(p, iris.cube.Cube) for p in products):
        return _multicube_statistics(
            cubes=products,
            statistics=statistics,
            span=span,
        )
    if all(type(p).__name__ == 'PreprocessorFile' for p in products):
        # Avoid circular input: https://stackoverflow.com/q/16964467
        return _multiproduct_statistics(
            products=products,
            statistics=statistics,
            output_products=output_products,
            span=span,
        )
    raise ValueError(
        "Input type for multi_model_statistics not understood. Expected "
        "iris.cube.Cube or esmvalcore.preprocessor.PreprocessorFile, "
        "got {}".format(products))
