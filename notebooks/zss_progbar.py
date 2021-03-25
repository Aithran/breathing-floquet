def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    from datetime import datetime, timedelta
    start_time = datetime.now() 
    
    clip_thresh = timedelta(seconds=10)
    
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            step_time = datetime.now()
            elapsed_time = (datetime.now() - start_time)
            elapsed_time -= timedelta(microseconds=elapsed_time.microseconds)
            elapsed_per = (elapsed_time/(index-1)) if index>1 else timedelta()
            elapsed_per_raw = elapsed_per
            if elapsed_per > clip_thresh:
                elapsed_per -= timedelta(microseconds=elapsed_per.microseconds)
            else:
                elapsed_per = '{:d}.{:03d} s'.format(elapsed_per.seconds, elapsed_per.microseconds//1000)
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ? ({elapsed}, {elapsed_per} per)'.format(
                            name=name,
                            index=index,
                            elapsed=elapsed_time,
                            elapsed_per=(elapsed_per if index>1 else "--")
                            )
                else:
                    progress.value = index
                    remaining = ((size - index + 1.0)*elapsed_time)/(index - 1.0) if index>1 else timedelta()
                    remaining -= timedelta(microseconds=remaining.microseconds)
                    label.value = u'{name}: {index} / {size} ({elapsed}, {elapsed_per} per, est. {pred_time} left)'.format(
                            name=name,
                            index=index,
                            size=size,
                            elapsed=elapsed_time,
                            elapsed_per=(elapsed_per if index>1 else "--"),
                            pred_time=(remaining if index>1 else "--")
                            )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        step_time = datetime.now()
        elapsed_time = (datetime.now() - start_time)
        elapsed_time -= timedelta(microseconds=elapsed_time.microseconds)
        elapsed_per = (elapsed_time/(index)) if index>1 else timedelta()
        if elapsed_per > clip_thresh:
            elapsed_per -= timedelta(microseconds=elapsed_per.microseconds)
        else:
            elapsed_per = '{:d}.{:03d} s'.format(elapsed_per.seconds, elapsed_per.microseconds//1000)
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index} ({elapsed}, {elapsed_per} per)".format(
                name=name,
                index=str(index or '?'),
                elapsed=elapsed_time,
                elapsed_per=(elapsed_per if index>1 else "--")
                )

def opt_log_progress(sequence, quiet=False, **kwargs):
    return sequence if quiet else log_progress(sequence, **kwargs)
