# Copyright (c) Microsoft Corporation.
# Licensed under the Apache 2.0 License.

"""
Relay program analysis framework: build human-readable analyses of Relay
programs.

TODO(gus) need better naming to differentiate from `analysis`.

For an example of how to use this framework, please read the unittests
in tests/python/unittest/test_analysis_tools.py.

How to use this framework:

1. Write an analysis by building a class that overrides AnalysisPass.
AnalysisPass is a thin wrapper over ExprVisitor, providing methods to
simplify the building of analyses. See AnalysisPass's docstring.

class MyAnalysis(AnalysisPass):
  def visit_call(self, call):
    ...perform analyses...

  def _summarize(self):
    ...summarize analyses...

2. Run analyses over your programs of interest using run_analyses.

module, _ = relay.testing.resnet.get_workload()
analysis, summary = run_analyses(module['main'],
                                 [MyAnalysis(),
                                  MyOtherAnalysis()])

3. Do useful stuff with the analyses using the provided utilities.
   - Print columns
   - Create an analysis writer
"""

from .expr_functor import ExprVisitor as _ExprVisitor
from .expr import Expr as _Expr
from typing import List, Dict


# TODO(gus) Analyses are usable only once, because they hold their data
# internally. Analysis passes should perhaps be viewed more as function
# descriptions that are re-entrant, returning their resulting data but
# not containing any state within them.
class AnalysisPass(_ExprVisitor):
    """Base class for building analysis passes.

    Analyses are simply thin wrappers over ExprVisitors, providing some
    additional useful functionality.


    ANALYSIS STRUCTURE

    This analysis framework enforces the following general analysis
    structure.

    Analyses are broken into two parts: the analysis proper, and the
    analysis summary.

    The analysis proper is best seen as a dictionary mapping nodes in
    the Relay program to dictionaries containing analysis data for each
    specific node. In this way, the analysis proper forms a table, where
    each node in the Relay program is a row, and each different type of
    analysis result a is a column. Please see the `statistics` property
    for a more detailed explanation of the analysis result format.

    The analysis summary, on the other hand, is a single dictionary not
    attached to any Relay node, but instead, containing summary
    information about the entire network. See the `summary` property for
    more information.

    BUILDING ANALYSES

    Building analyses involves two steps, both of which are optional:
    1. Building the analysis proper by attaching analysis results to
       nodes in the program.
    2. Building the summary.

    To build the analysis proper, the user will attach analysis results
    to nodes in the program. This is implemented by AnalysisPass using
    the visitor pattern employed by ExprVisitor. When building an
    analysis pass, the user may override the visitor functions (such as
    `visit_var()`). Inside those functions, the user will generate
    analysis results, and then attach those results to the node using
    `_add_detail()`.

    Most likely, the user will want to override `visit_call`, to visit
    the different call nodes of the program. To make analyses cleaner,
    AnalysisPass allows the user to visit specific calls by overriding
    `visit_call_<op name>`, where dots (.) in the operator name are
    replaced with underscores (_). For example, if our analysis only
    needed to visit calls to `nn.conv2d`, we would write:

      def visit_call_nn_conv2d(self, conv2d):
        some_info = conv2d.attrs[...]
        some_other_info = conv2d.checked_type.shape[...]
        result = ...analysis using some_info and some_other_info...
        self._add_detail(something_about_conv2d=result)

    If the AnalysisPass is initialized with `warn_unimplemented=True`,
    any `visit_call_...` not implemented by your analysis pass will
    cause a warning. This is useful when you would like to implement
    a visitor for each operator type.

    See the docstring of `visit_call()` for more info.

    The analysis can rely on existing data from other analysis passes.
    Existing data is made available via `self._existing_data`. There is
    currently no mechanism to enforce dependencies between analyses. If
    your analysis pass A depends on another pass B, and you forget to
    run that pass, you will encounter errors (likely in the form of
    `KeyError`s on `self._existing_data`).

    To build the summary, the user will override `_summarize()`. This
    function will be run after the analysis proper finishes running over
    all nodes. All of the analysis data generated is visible via the
    `statistics` property. Finally, the summary data can be added to the
    results using `_add_summary()`.

    The results of the analysis proper and the analysis summary can be
    accessed via `statistics` and `summary`, respectively.
    """

    def __init__(self, warn_unimplemented=False):
        """
        warn_unimplemented: warn if we encounter `Call`s for which a
        `visit` is not implemented."""
        super().__init__()
        self._exprs_to_data = {}
        self._warn_unimplemented = warn_unimplemented
        # Maps strings (column names) to summary data
        self._summary_data = {}


    @property
    def statistics(self) -> Dict[_Expr, Dict]:
        """Statistics produced by this analysis pass

        The results of an analysis is a dictionary mapping nodes in
        the Relay program to analysis "data dictionaries" containing
        analysis results about each node. We assume the data
        dictionaries attached to each node use strings as their keys.
        These data dictionaries can be nested. For example:
        {
        <relay node> : {
                        'name' : 'nn.conv2d',
                        'filter_info' : {
                            'filter_size' : ...,
                            'channels' : ...
                        }
                        },
        <another relay node> : {
                                'name' : 'nn.dropout',
                                'dropout_rate' : ...
                                }
        }
        Analysis results can also be seen as a table, with a row for
        each Relay node and a column for each unique path of keys across
        all of the data dictionaries. In the examples above, we would
        have two rows (one for each of the Relay nodes) and four columns
        representing the following column paths:
        - ('name',)
        - ('filter_info', 'filter_size')
        - ('filter_info', 'channels')
        - ('dropout_rate',)
        """
        return self._exprs_to_data


    @property
    def summary(self):
        """Summary produced by this analysis pass

        The summary is an analysis result not attached to any specific
        node in the Relay program. It is a single data dictionary not
        attached to any Relay node, but instead, containing analysis
        information pertaining to the network as a whole. For example:
        {
        'op_histogram' : {
                            'nn.conv2d' : ...,
                            'nn.dropout' : ...,
                            ...
                        },
        'input_size' : ...
        }
        While the results of the analysis proper form a spreadsheet with
        one row per Relay node, the analysis summary can be seen as a
        single row in a spreadsheet. In the example above, the columns
        of this row would be
        - ('op_histogram', 'nn.conv2d')
        - ('op_histogram', 'nn.dropout')
        - ('op_histogram', ...)
        - ...
        - ('input_size', )
        A summary is useful in a few instances. First, when collecting
        metrics about the network as a whole, which can't necessarily be
        attached to a single node---for example, a histogram of all
        operators. Second, summaries can be especially useful when
        analyzing multiple Relay programs, as it allows the user to
        produce a summary sheet with a single row per program, for easy
        comparison of high-level metrics between programs.
        """
        return self._summary_data


    # TODO(gus) existing data different from exprs_to_data? That
    # doesn't seem right.
    def _run(self, expr: _Expr, existing_data: Dict[_Expr, Dict]):
        """For internal use: run the analysis pass.

        *NOTE:* Users of the analysis framework should use run_analyses,
        rather than running analyses manually using this method.

        This method runs the analysis, first taking in any data that
        might have been produced by previous passes.

        Users should prefer run_analyses, as it handles the passing of
        analysis data (ie existing_data) between analyses.

        expr: the expression to run the analysis on.
        existing_data: data produced by previous analyses.
        returns: data produced by this analysis, in the same format as
                 existing_data.
        """
        self._existing_data = existing_data
        super().visit(expr)
        self._summarize()

    def visit_call(self, c):
        """
        This implementation of `visit_call` increases the granularity of
        `ExprVisitor`'s visitor pattern. Instead of needing to switch
        over the different `c.op.name`s (e.g. "nn.conv2d" or "add"),
        this override allows the user to implement
        `visit_call_<op name>()`, where any periods are replaced with
        underscores. For example, to visit calls to `nn.conv2d`, one
        would implement `visit_call_nn_conv2d(self, conv2d)`.

        TODO(gus) currently, there's no clean way to override visit_call
        without getting a ton of warnings from not implementing the
        methods for each call type.
        """
        super().visit_call(c)

        try:
            name = f'visit_call_{c.op.name.replace(".","_")}'
        except AttributeError as e:
            name = f'visit_call_{c.op.name_hint.replace(".","_")}'

        try:
            getattr(self, name)(c)
        except AttributeError as e:
            if name not in str(e):
                raise e
            elif self._warn_unimplemented:
                import sys
                sys.stderr.write(f'Unimplemented: {e}\n')

    def _add_detail(self, expr: _Expr, data: dict = {}, **kwargs):
        """Add analysis details for an expression in the Relay program

        `expr`: the `relay.Expr` which these analysis results pertain to
        `data`: dictionary of analysis results. See below for more
                details on the format. See `statistics` also.
        `kwargs`: syntactic sugar for specifying a `data` dictionary.
                  Calling `_add_detail(expr, key1=value1, key2=value2)`
                  is equivalent to
                  `_add_detail(expr, {key1 : value1, key2 : value2})`.

        The data dictionary is expected to be in the following format:
        ```
        data = {
          key1 : value1,
          key2 : {
            key3 : value3,
            key4 : { ... },
            ...
          },
          ...
        }
        ```
        All keys are strings. Values can be nested dictionaries or
        analysis results.

        Example:
        _add_detail(call, {
          'type_info' : {
            'dtype' : expr.checked_type.dtype,
            'shape' : expr.checked_type.shape,
          },
          'name' : call.op.name,
        })
        which is equivalent to:
        _add_detail(call,
          type_info = {
            'dtype' : expr.checked_type.dtype,
            'shape' : expr.checked_type.shape,
          },
          name=call.op.name)
        """
        if expr not in self._exprs_to_data: self._exprs_to_data[expr] = {}
        self._exprs_to_data[expr].update(data)
        self._exprs_to_data[expr].update(kwargs)

    def _add_summary(self, summary):
        """Set the summary for this analysis."""
        self._summary_data = summary


    def _summarize(self):
        """This function runs after `run` completes.

        This function can be overridden and used as a convenient
        location for generating the analysis summary."""
        pass


def run_analyses(expr, passes: List[AnalysisPass]) -> Dict[_Expr, Dict]:
    """Run a set of analyses and return the results

    This is the only function which should be used to run analyses.
    Users should NOT explicitly call an analysis's `_run()` method.
    This function handles the passing of data between analysis
    passes in a transparent way.

    expr: the expression to run on.
    passes: the passes to be run on the expression."""
    out = {}
    summaries = {}
    for _pass in passes:
        _pass._run(expr, out)

        # Merge results into `out`
        for _expr in _pass.statistics:
            if _expr in out:
                __merge(out[_expr], _pass.statistics[_expr])
            else: out[_expr] = _pass.statistics[_expr]

        __merge(summaries, _pass.summary)
    return (out, summaries)


def __merge(a: Dict, b: Dict):
    """Merge two generic dictionaries in-place

    Merges b into a.

    This function will throw an assertion error if the same field is set
    in both dictionaries, e.g.
    {
      ...
      'a' : {
        'b' : 1
      }
      ...
    }
    and
    {
      ...
      'a' : {
        'b' : 2
      }
      ...
    }
    """
    for a_key in a.keys():
        if a_key in b:
            assert isinstance(a_val, dict) and isinstance(b[a_key], dict)
            __merge(a[a_key], b[a_key])
    for b_key, b_val in b.items():
        if b_key in a:
            # TODO(gus) check that it's correctly merged in?
            pass
        else:
            # Stick all the other ones into a.
            a[b_key] = b_val


def _dictionary_to_column_paths(dictionary, prefix=tuple()):
    """Convert a dictionary to the column paths within this dictionary

    For example, if the argument is
    {
      1 : {
            'a' : True,
            'b' : False
          },
      (10, 'blah') : SomeObject()
    }

    The result would be
    [
      (1, 'a'),
      (1, 'b'),
      ((10, 'blah'))
    ]
    """
    paths = set()
    for key, val in dictionary.items():
        if isinstance(val, dict):
            paths.update(_dictionary_to_column_paths(val, prefix + (key,)))
        else:
            paths.add(prefix + (key,))
    return paths


def get_summary_columns(summary):
    return _dictionary_to_column_paths(summary)


def get_analysis_columns(analysis):
    analysis_columns = set()
    for analysis_data in analysis.values():
        analysis_columns.update(_dictionary_to_column_paths(analysis_data))
    return analysis_columns


def get_columns(data: Dict[_Expr, Dict]):
    """For this set of data, get the column structure

    The resulting structure is a dictionary mapping column identifiers"""

    def __update_columns(columns: Dict, data: Dict):
        for key, val in data.items():
            if isinstance(val, dict):
                # create dict if doesn't exist
                if key not in columns: columns[key] = {}
                __update_columns(columns[key], val)
            else:
                columns[key] = None

    columns = {}
    for expr, values in data.items():
        __update_columns(columns, values)
    return columns


def print_analyses_in_spreadsheet(data: Dict[_Expr, Dict], print_columns, cell, end_of_row):
    """
    print_columns: lambda taking a list of column identifiers or nested
    lists. the DFS order of the columns is how the columns should be
    ordered in the sheet.
    print_row: lambda taking the expr, row of data, and column structure."""

    columns = get_columns(data)

    column_to_index = {}
    def dfs(columns: Dict, prefix: tuple, i: int):
        for key, val in columns.items():
            if isinstance(val, dict):
                dfs(val, prefix + (key, ), i)
            else:
                column_to_index[prefix + (key,)] = i
                i += 1
    dfs(columns, (), 0)

    index_to_column = {val : key for key, val in column_to_index.items()}
    indices = sorted(list(index_to_column.keys()))

    print_columns(index_to_column)

    for expr, data in data.items():
        # Go through columns in order
        for index in indices:
            # Find value if it exists, or find None otherwise.
            column_tuple = index_to_column[index]
            val = data
            for column in column_tuple:
                if column in val:
                    val = val[column]
                else:
                    val = None
                    break
            cell(data, val, index)
        end_of_row()


def print_analyses_v2(data: Dict[_Expr, Dict], column_order, cell_callback, end_of_row_callback, spreadsheet_data=None):
    """
    column_order: list of tuples of column-group/column ids, specifying
    path to leaf column. Any column not listed here will not be printed.
    This specifies the order in which data will be printed.
    """

    for expr, data in data.items():
        # Go through columns in order
        for i, column_tuple in enumerate(column_order):
            val = data
            for column in column_tuple:
                if column in val:
                    val = val[column]
                else:
                    val = None
                    break
            spreadsheet_data = cell_callback(expr, i, val, spreadsheet_data)
        spreadsheet_data = end_of_row_callback(spreadsheet_data)

    return spreadsheet_data

def get_records(data: Dict[_Expr, Dict], column_order):
    """Output data as a list of records"""
    def cell_callback(expr, i, val, spreadsheet_data):
        spreadsheet_data[-1].append(val)
        return spreadsheet_data
    def row_callback(spreadsheet_data):
        spreadsheet_data[-1] = tuple(spreadsheet_data[-1])
        spreadsheet_data.append([])
        return spreadsheet_data
    out = [[]]
    out = print_analyses_v2(data, column_order, cell_callback, row_callback, out)
    return out[:-1]

def summary_to_record(paths, summary):
    """Turn summary results into a record.

    paths: a list of column paths, where column paths are tuples
    of strings specifying the path to each column.
    summary: the model's summary produced by run_analyses."""

    record_list = []
    for path in paths:
        val = summary
        for path_part in path:
            if path_part in val:
                val = val[path_part]
            else:
                val = None
                break
        record_list.append(val)
    return tuple(record_list)
