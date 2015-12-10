from plotly import session, tools, utils
import uuid
import json
import mu2e
import os
from pkg_resources import resource_string



def new_iplot(figure_or_data, show_link=True, link_text='Export to plot.ly',
          validate=True):

    figure = tools.return_figure_from_figure_or_data(figure_or_data, validate)

    width = figure.get('layout', {}).get('width', '100%')
    height = figure.get('layout', {}).get('height', 525)
    try:
        float(width)
    except (ValueError, TypeError):
        pass
    else:
        width = str(width) + 'px'

    try:
        float(width)
    except (ValueError, TypeError):
        pass
    else:
        width = str(width) + 'px'

    plotdivid = uuid.uuid4()
    jdata = json.dumps(figure.get('data', []), cls=utils.PlotlyJSONEncoder)
    jlayout = json.dumps(figure.get('layout', {}), cls=utils.PlotlyJSONEncoder)

    config = {}
    config['showLink'] = show_link
    config['linkText'] = link_text
    jconfig = json.dumps(config)

    plotly_platform_url = session.get_session_config().get('plotly_domain',
                                                           'https://plot.ly')
    if (plotly_platform_url != 'https://plot.ly' and
            link_text == 'Export to plot.ly'):

        link_domain = plotly_platform_url\
            .replace('https://', '')\
            .replace('http://', '')
        link_text = link_text.replace('plot.ly', link_domain)


    script = '\n'.join([
        'Plotly.plot("{id}", {data}, {layout}, {config}).then(function() {{',
        '    $(".{id}.loading").remove();',
        '}})'
    ]).format(id=plotdivid,
              data=jdata,
              layout=jlayout,
              config=jconfig)

    html="""<script src="../plotly.min.js"></script>
    <div class="{id} loading" style="color: rgb(50,50,50);">
                 Drawing...</div>
                 <div id="{id}" style="height: {height}; width: {width};"
                 class="plotly-graph-div">
                 </div>
                 <script type="text/javascript">
                 {script}
                 </script>
                 """.format(id=plotdivid, script=script,
                           height=height, width=width)

    return html

def get_plotlyjs():
    path = os.path.dirname(mu2e.__file__)+'/tools/plotly.min.js'
    with open(path) as myfile:
        plotlyjs = myfile.read().decode('utf-8')
    return plotlyjs

    #return plotlyjs
#def getCustomJS():
#        # plotlyjs.js is the javascript file from the plotly repo
#    with open (os.path.dirname(mu2e.__file__)+"/tools/plotly.js", "r") as myfile:
#        data=myfile.read().replace('\n', '')
#    return data
