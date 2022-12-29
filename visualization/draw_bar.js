var myChart = echarts.init(dom, null, {
    renderer: "svg",
    useDirtyRect: false,
    width: 40 + 70 * 3,
    height: 200
  });
  

  
option = {
    legend: {},
    tooltip: {},
    grid: {
      left: 30,
      right: 3,
      bottom: 20
    },
    dataset: {
      source: [
        ["Name", 'GATv2', 'DPGAT'],
        ['n=20', 99.9, 43.3],
        ['n=30', 99.9, 43.4],
        ['n=40', 99.9, 43.3],
      ]
    },
    xAxis: { type: 'category' },
    yAxis: {},
    series: [{ 
      type: 'bar',
      label: {
        position: [0, -14],
        show: true
      }
    }, { 
      type: 'bar',
      label: {
        position: [0, -14],
        show: true
      }
    }]
  };


  option = {
    legend: {},
    tooltip: {},
    grid: {
      left: 30,
      right: 3,
      bottom: 20
    },
    dataset: {
      source: [
        ["Name", 'GATv2', 'DPGAT'],
        ['n=20', 99.9, 84.1],
        ['n=30', 99.9, 84.0],
        ['n=40', 99.9, 84.1],
        ['n=100',99.9, 85.0]
      ]
    },
    xAxis: { type: 'category' },
    yAxis: {},
    series: [{ 
      type: 'bar',
      label: {
        position: [0, -14],
        show: true
      }
    }, { 
      type: 'bar',
      label: {
        position: [0, -14],
        show: true
      }
    }]
  };

  option = {
    legend: {},
    tooltip: {},
    grid: {
      left: 30,
      right: 3,
      bottom: 20
    },
    dataset: {
      source: [
        ["Name", 'GATv2', 'DPGAT'],
        ['n=20',  99.9, 98.7],
        ['n=30',  99.9, 98.6],
        ['n=40',  99.9, 98.6],
        ['n=100', 99.9, 98.4],
        ['n=200', 99.9, 84.7],
        ['n=500', 99.9, 84.6],
        ['n=1000',99.9, 84.6]
      ]
    },
    xAxis: { type: 'category' },
    yAxis: {},
    series: [{ 
      type: 'bar',
      label: {
        position: [0, -14],
        show: true
      }
    }, { 
      type: 'bar',
      label: {
        position: [0, -14],
        show: true
      }
    }]
  };
