/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.elasticsearch.search;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.*;
import org.apache.lucene.queries.function.FunctionScoreQuery;
import org.apache.lucene.search.*;
import org.apache.lucene.util.BytesRef;
import org.elasticsearch.ElasticsearchException;
import org.elasticsearch.analysis.IvfpqAnalyzer;
import org.elasticsearch.ann.ArrayUtils;
import org.elasticsearch.ann.ExactSearch;
import org.elasticsearch.ann.ProductQuantizer;
import org.elasticsearch.index.analysis.NamedAnalyzer;
import org.elasticsearch.index.mapper.MappedFieldType;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.MatchAllQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryShardContext;
import org.elasticsearch.mapper.IvfpqFieldMapper;

import java.io.IOException;
import java.util.*;

class IvfpqQuery {

    public static final Logger LOGGER = LogManager.getLogger(IvfpqQuery.class);

    private QueryShardContext context;

    IvfpqQuery(QueryShardContext context) {
        this.context = context;
    }

    Query parse(QueryBuilder in, Map<String, Float> fieldNames, Object value, int nprobe) throws IOException {
        float[] features = ArrayUtils.parseFloatArrayCsv((String) value);
        //List<Query> fieldQueries = new ArrayList<>();
        Query query = null;
        long time1 = System.currentTimeMillis();
        for (String field : fieldNames.keySet()) {
            MappedFieldType fieldMapper = context.fieldMapper(field);
            Analyzer analyzer = context.getSearchAnalyzer(fieldMapper);
            while (analyzer instanceof NamedAnalyzer) {
                analyzer = ((NamedAnalyzer) analyzer).analyzer();
            }
            if (!(analyzer instanceof IvfpqAnalyzer)) {
                throw new ElasticsearchException("illegal analyzer: " + analyzer);
            }
            ProductQuantizer pq = ((IvfpqAnalyzer) analyzer).getProductQuantizer();
            ExactSearch cq = ((IvfpqAnalyzer) analyzer).getCoarseQuantizer();
            Map<String, float[]> tables = new HashMap<>();
            List<BytesRef> nearCode = new ArrayList<>();
            for (int nearest : cq.searchNearest(features, nprobe)) {
                float[] residual = cq.getResidual(nearest, features);
                float[] table = pq.getCodeTable(residual);
                String code = String.valueOf(nearest);
                nearCode.add(new BytesRef(code));
                tables.put(code, table);
            }
            BooleanQuery.Builder builder = new BooleanQuery.Builder().
                    add(new TermInSetQuery(field, nearCode), BooleanClause.Occur.FILTER);
            if (!(in instanceof MatchAllQueryBuilder)) {
                builder.add(in.toQuery(context), BooleanClause.Occur.FILTER).build();
            }
            query = new FunctionScoreQuery(builder.build(),
                    new CustomValueSource(field, pq, tables));
        }
        return query;
    }

    private class CustomValueSource extends DoubleValuesSource {

        private String field;

        private ProductQuantizer pq;

        private Map<String, float[]> codeTable;

        CustomValueSource(String field, ProductQuantizer pq, Map<String, float[]> codeTable) {
            this.field = field;
            this.pq = pq;
            this.codeTable = codeTable;
        }

        @Override
        public DoubleValues getValues(LeafReaderContext leafReaderContext, DoubleValues scores) {
            return new CustomDoubleValues(leafReaderContext);
        }

        public class CustomDoubleValues extends DoubleValues {

            private float value;

            private BinaryDocValues ivfDocValues;

            private BinaryDocValues pgDocValues;

            public CustomDoubleValues(LeafReaderContext leafReaderContext) {
                try {
                    ivfDocValues = DocValues.getBinary(leafReaderContext.reader(), field);
                    pgDocValues = DocValues.getBinary(leafReaderContext.reader(), IvfpqFieldMapper.getCodesField(field));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            @Override
            public double doubleValue() {
                return value;
            }

            @Override
            public boolean advanceExact(int doc) throws IOException {
                ivfDocValues.advanceExact(doc);
                pgDocValues.advanceExact(doc);
                byte[] pqCode = pgDocValues.binaryValue().bytes;
                String nearest = ivfDocValues.binaryValue().utf8ToString();
                if (pqCode == null || nearest == null) {
                    return false;
                }
                short[] codes = ArrayUtils.decodeShortArray(pqCode);
                value = pq.getDistance(codeTable.get(nearest), codes);
                return true;
            }
        }

        @Override
        public boolean needsScores() {
            return false;
        }

        @Override
        public DoubleValuesSource rewrite(IndexSearcher reader) {
            return this;
        }

        @Override
        public boolean equals(Object o) {
            if (o instanceof IvfpqQuery) {
                return o.hashCode() == this.hashCode();
            }
            return false;
        }

        @Override
        public int hashCode() {
            return Objects.hash(field, codeTable);
        }

        @Override
        public String toString() {
            return null;
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
            return false;
        }
    }

}
