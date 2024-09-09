for i in *md;
do
	outF=${i/md/html}
	titN=${i/.md/}
	echo $outF
	cat header.html | sed 's/title_name/'"$titN"'/' > $outF
        pandoc -f markdown+pipe_tables $i >> $outF
	echo -e  '</body></html>' >> $outF
done


