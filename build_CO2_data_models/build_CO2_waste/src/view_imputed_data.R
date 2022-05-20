library(ggplot2)
library(reshape)

df <- read.csv("/home/milo/Documents/egap/descarbonizacion/data_imputation/data/waste_ws_datos_completos.csv")
column_names_changes <- stringr::str_replace_all(tolower(unique(df$Country))," ","_")
column_names_original <- unique(df$Country)

for (i in c(1:length(column_names_original))) {
  df$Country[df$Country==column_names_original[i]] <-column_names_changes[i]

}

colnames(df) <- c("year","country","value")
df <- df[,c("year","value","country")]
df$cat <- "observado"
df_imputed <- read.csv("/home/milo/Documents/egap/descarbonizacion/data_imputation/output/waste_ws_datos_imputados.csv")

df_imputed<- subset(df_imputed,category=="imputed")
df_imputed <- df_imputed[,-4]
df_imputed$cat = "imputado"

df_completo <- rbind.data.frame(df,df_imputed)
column_names <- unique(df_imputed$country)

for (c in column_names) {
    print(c)
    df_observado <- subset(df_completo,country==c & cat=="observado")
    df_imputado <- subset(df_completo,country==c & cat=="imputado")

    ggplot(df_imputado, aes(x = year, y = value))+geom_line()+geom_point(aes(x=year, y=value), size=3,df_observado,color="red")+
      labs(title=paste0(c," pronosticado vs observado"),x ="Time", y = "Total CO2 emission")+ theme_minimal()
    #ggsave(paste0("/home/milo/Documents/egap/descarbonizacion/data_imputation/output/images/",country,"_imputation.png"), bg = "white")
    ggsave(paste0("/home/milo/Documents/egap/descarbonizacion/data_imputation/output/images/",c,"_imputation.eps"), bg = "white",device = cairo_ps)
}

 gg <- ggplot()
 gg+ geom_point(df_observado, aes(x=year, y=value), size=5, alpha=0.3)+ geom_line(df_imputado, aes(x=year, y=value))
