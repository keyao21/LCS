from ftle import FTLE

def test1():
    datafolder = '../data'
    for length in range(60, 90, 10):
        # print(length, int(length/2))
        mappingfile = '{}/kevin_output_steps.txt'.format(datafolder)
        test = FTLE(length=length, height=int(length/2))
        test.createMapping(mappingfile)
        test.plotter(display=False)
        print( '{}, {}x{}'.format(test.total_time, test.length, test.height ) )

def test2():
    datafolder = '../data'
    mappingfile = '{}/test2.txt'.format(datafolder)
    test = FTLE(length=100, height=50)
    test.createMapping(mappingfile)
    test.plotter()

def testDiscrete():
    datafolder = '../data'
    mappingfile = '{}/testDiscrete.txt'.format(datafolder)

    test = FTLE(length=100, height=50, estimateVelocity=True)
    test.createMapping(mappingfile)
    test.plotter(display=False, title='With discretized velocity field', filename='discrete.pdf')
    
    test = FTLE(length=100, height=50, estimateVelocity=False)
    test.createMapping(mappingfile)
    test.plotter(display=False, title='With continuous velocity field', filename='continuous.pdf')

def testDiscrete2():
    datafolder = '../data'
    mappingfile = '{}/testDiscrete.txt'.format(datafolder)

    test = FTLE(length=100, height=50, percentLength=0.5, percentHeight=0.5, estimateVelocity=True)
    test.createMapping(mappingfile)
    test.plotter(display=true, title='test', filename='TESTS')
    
def main():
    testDiscrete2()


if __name__ == '__main__':
    main()