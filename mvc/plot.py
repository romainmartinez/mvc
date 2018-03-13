import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.tools as tools

BASE_LAYOUT = go.Layout(hovermode='closest', font=dict(size=14))
MARKER_LAYOUT = dict(
    color='rgba(27, 158, 119, 0.6)',
    line=dict(
        color='rgba(27, 158, 119, 1.0)',
        width=2,
    ))


def count_by_dataset(d, values, index, columns, **kwargs):
    table = d.pivot_table(
        values,
        index,
        columns,
        aggfunc=lambda x: len(x) / x.nunique(),
        fill_value=0).astype(int)

    fig = ff.create_annotated_heatmap(
        z=np.array(table),
        x=kwargs.get('xlabel'),
        y=kwargs.get('ylabel'),
        showscale=True,
        colorscale='YlGnBu',
        colorbar=dict(title='Count', titleside='right'))

    fig['layout'].update(BASE_LAYOUT)
    fig['layout'].update(
        dict(
            title=kwargs.get('title'),
            xaxis=dict(title=kwargs.get('xtitle'), side='bottom'),
            yaxis=dict(title=kwargs.get('ytitle'), autorange='reversed'),
            margin=go.Margin(t=80, b=80, l=150, r=80, pad=0)))
    return fig


def count_bar(d, column, **kwargs):
    count = np.array(d[column].value_counts(sort=False))
    trace = go.Bar(
        x=count, y=kwargs.get('ylabel'), marker=MARKER_LAYOUT, orientation='h')

    layout = BASE_LAYOUT.copy()
    layout.update(
        dict(
            title=kwargs.get('title'),
            xaxis=dict(
                title=kwargs.get('xtitle'), showline=True, linewidth=1.5),
            yaxis=dict(
                title=kwargs.get('ytitle'), showline=True, linewidth=1.5)))

    # adjust y axis
    layout['yaxis'].update(nticks=count.shape[0])
    layout.update(margin=go.Margin(t=80, b=80, l=150, r=80, pad=0))
    return dict(data=[trace], layout=layout)


def max_by_test(d, **kwargs):
    maximum = d[d['mvc'] == 100].pivot_table(
        values='mvc',
        index='muscle',
        columns='test',
        aggfunc='count',
        fill_value=0)
    maximum = (maximum.div(maximum.sum(axis=1), axis=0) * 100).astype(int)

    fig = ff.create_annotated_heatmap(
        z=np.array(maximum),
        x=kwargs.get('xlabel'),
        y=kwargs.get('ylabel'),
        showscale=True,
        colorscale='YlGnBu',
        colorbar=dict(title='Percentage', titleside='right'))

    fig['layout'].update(BASE_LAYOUT)
    fig['layout'].update(
        dict(
            title=kwargs.get('title'),
            xaxis=dict(title=kwargs.get('xtitle'), side='bottom'),
            yaxis=dict(title=kwargs.get('ytitle'), autorange='reversed'),
            margin=go.Margin(t=80, b=80, l=150, r=80, pad=0)))
    return fig


def count_nan(d, **kwargs):
    nan_count = d.isnull().sum()
    nan_id = nan_count.index

    trace = go.Bar(x=nan_id, y=nan_count, marker=MARKER_LAYOUT)
    layout = BASE_LAYOUT.copy()
    layout.update(
        dict(
            title=kwargs.get('title'),
            xaxis=dict(
                title=kwargs.get('xtitle'), showline=True, linewidth=1.5),
            yaxis=dict(
                title=kwargs.get('ytitle'), showline=True, linewidth=1.5)))

    annotations = []
    for count, idx in zip(nan_count, nan_id):
        annotations.append(
            dict(
                y=count + 50,
                x=idx,
                text=count,
                font=dict(size=14),
                showarrow=False))
        if count < 50:
            annotations.append(
                dict(
                    y=count + 100,
                    x=idx,
                    text=f'test {idx}',
                    font=dict(size=14),
                    showarrow=True,
                    arrowhead=7,
                    ax=0,
                    ay=-200))
    layout['annotations'] = annotations
    return dict(data=[trace], layout=layout)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def mvc_positions(positions, idx, **kwargs):
    n_rows = int(len(idx) / 4) + 1
    n_cols = 4
    height = n_rows * 800 / 4
    width = n_cols * 800 / 4

    fig = tools.make_subplots(
        rows=n_rows,
        cols=n_cols,
        print_grid=False,
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
        subplot_titles=[f'MVC {i}' for i in idx])

    for i, itest in enumerate(idx):
        fig.append_trace(
            go.Heatmap(
                z=rgb2gray(positions[0][itest]),
                colorscale='Greys',
                showscale=False,
                name=f'MVC {itest}'),
            row=int(i / n_cols) + 1,
            col=int(i % n_cols) + 1)
        fig['layout']['title'] = kwargs.get('title')
        fig['layout'].update(height=height, width=width)
        fig['layout'][f'xaxis{i+1}'].update(showticklabels=False, ticks='')
        fig['layout'][f'yaxis{i+1}'].update(
            autorange='reversed', showticklabels=False, ticks='')
    return fig


def learning_curve(estimator,
                   X,
                   y,
                   scoring,
                   random_state,
                   cv=None,
                   n_jobs=1,
                   train_sizes=np.linspace(.1, 1.0, 5),
                   **kwargs):
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        random_state=random_state,
        scoring=scoring,
        cv=cv,
        train_sizes=train_sizes,
        n_jobs=n_jobs)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    trace = []

    # training mean
    trace.append(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            marker=dict(color='red'),
            name='Training score'))

    # training std
    trace.append(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean + train_scores_std,
            mode='lines',
            line=dict(color='red', width=1),
            showlegend=False))

    trace.append(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean - train_scores_std,
            mode='lines',
            line=dict(color='red', width=1),
            fill='tonexty',
            showlegend=False))

    # test mean
    trace.append(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            marker=dict(color='green'),
            name='Cross-validation score'))

    # test std
    trace.append(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean + test_scores_std,
            mode='lines',
            line=dict(color='green', width=1),
            showlegend=False))

    trace.append(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean - test_scores_std,
            mode='lines',
            line=dict(color='green', width=1),
            fill='tonexty',
            showlegend=False))

    data = [itrace for itrace in trace]
    layout = BASE_LAYOUT.copy()
    layout.update(
        dict(
            title=kwargs.get('title'),
            xaxis=dict(
                title=kwargs.get('xtitle'), showline=True, linewidth=1.5),
            yaxis=dict(
                title=kwargs.get('ytitle'), showline=True, linewidth=1.5)))
    fig = dict(data=data, layout=layout)
    return fig


def bar_metrics(d, **kwargs):
    trace_mape = go.Bar(
        x=np.array(d.index),
        y=np.array(d['mape']),
        marker=MARKER_LAYOUT,
        name='mape')

    trace_rmse = go.Scatter(
        x=np.array(d.index),
        y=np.array(d['rmse']),
        marker=dict(
            color='rgba(117, 112, 179, 0.6)',
            line=dict(
                color='rgba(117, 112, 179, 1.0)',
                width=2,
            )),
        name='rmse',
        xaxis='x1',
        yaxis='y2')

    traces = [trace_mape, trace_rmse]

    layout = BASE_LAYOUT.copy()
    layout.update(
        dict(
            title=kwargs.get('title'),
            xaxis=dict(
                title=kwargs.get('xtitle'), showline=True, linewidth=1.5),
            yaxis=dict(
                title=kwargs.get('ytitle'), showline=True, linewidth=1.5),
            yaxis2=dict(
                overlaying='y',
                side='right',
                showgrid=False,
                title=kwargs.get('y2title'),
                showline=True,
                zeroline=False,
                linewidth=1.5),
            legend=dict(x=1, y=1.1)))
    return dict(data=traces, layout=layout)


def regression_report(d):
    table = ff.create_table(d, index=[d.index])

    table['layout'].update(font=dict(size=14))
    return table
