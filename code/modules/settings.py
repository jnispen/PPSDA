import datetime

def save(filepath, configuration):
    """ save the local configuration """
    """ parameters:
            filepath      = pathname to output file
            configuration = dictionary containing the local configuration
    """
    fp = open(filepath + '/settings.txt', 'w')
    fp.write('start       = ' + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p") + '\n')
    fp.write('scenario    = ' + configuration['scenario'] + '\n')
    fp.write('init_mode   = ' + configuration['init_mode'] + '\n')
    fp.write('prior_model = ' + configuration['prior_model'] + '\n')
    fp.write('peak_info   = ' + configuration['peak_info'] + '\n')
    if 'peak_shift' in configuration:
        fp.write('peak_shift  = ' + str(configuration['peak_shift']) + '\n')
    if 'model_mode' in configuration:
        fp.write('model_mode  = ' + configuration['model_mode'] + '\n')
    if 'data_mode' in configuration:
        fp.write('data_mode   = ' + configuration['data_mode'] + '\n')
    if 'dataset_dir' in configuration:
        fp.write('data_dir    = ' + configuration['dataset_dir'] + '\n')
    if 'niter' in configuration:
        fp.write('niter       = ' + str(configuration['niter']) + '\n')
    if 'nruns' in configuration:
        fp.write('nruns       = ' + str(configuration['nruns']) + '\n')
    fp.write('ncores      = ' + str(configuration['ncores']) + '\n')
    fp.write('nsamples    = ' + str(configuration['nsamples']) + '\n')
    fp.close()

def close(filepath):
    fp = open(filepath + '/settings.txt', 'a')
    fp.write('stop        = ' + datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p") + '\n')
    fp.close()
