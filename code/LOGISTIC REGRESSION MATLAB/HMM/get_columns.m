function columns = get_columns(k,i,j)
    %multiple columns output
    if nargin<3
        %no j provided, all j=1,...,k considered
        columns = [];
        for j=1:k
            columns = [columns, base2dec([num2str(i-1) num2str(j-1)],k)+1];
        end
    else %only one column output
        columns = base2dec([num2str(i-1) num2str(j-1)],k)+1;
    end
end