function data = np_matlab(np_data)

    shape = cellfun(@int64,cell(np_data.shape));
    ls = py.array.array('d', np_data.flatten('F').tolist());
    p = double(ls);

    data = reshape(p,shape);
end