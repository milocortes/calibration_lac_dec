library(ggplot2)

df <- read.csv("/home/milo/Documents/egap/descarbonizacion/data_imputation/data/waste_ws_datos_completos.csv")

# subset 1990-2016
# Multiple line plot
ggplot(df, aes(x = Year, y = Value)) +
  geom_line(aes(color = Country), size = 1) +
  theme_minimal()

column_names <- stringr::str_replace_all(tolower(unique(df$Country))," ","_")
df_cast <-cast(df,Year~Country)
df_cast <- df_cast[,-1]
colnames(df_cast)<-column_names

df_cast  <- df_cast[,!column_names %in% c("suriname","trinidad_and_tobago","venezuela_(bolivarian_republic_of)")]
write.csv(df_cast,"/home/milo/Documents/egap/descarbonizacion/data_imputation/data/waste_ts_data.csv",row.names=FALSE)
