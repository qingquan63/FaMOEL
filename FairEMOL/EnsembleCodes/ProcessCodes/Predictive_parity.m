function [Res, detail] = Predictive_parity(Preds, idxs_sensitive, Truelabel, num_group)
    
    Preds = Preds(:);
    Truelabel = double(Truelabel(:));
    Preds(Preds >= 0.5) = 1;
    Preds(Preds < 0.5) = 0;
    res = [];
    detail = [];
    for g = num_group
        gr = idxs_sensitive == g;
        truelabel = Truelabel(gr);
        preds = Preds(gr);
        if sum(preds) > 0
            res = [res sum((truelabel) .*  (preds)) / sum(preds)];
        end
        detail = [detail, sum((truelabel) .*  (preds)) ,  sum(preds)];
    end
    
    Res = get_ave_metrics(res, 1);
end

