library(ggplot2)

args = commandArgs(trailingOnly=TRUE)

sector.model <- args[1]

wd <- getwd()

if(sector.model=="AFOLU"){
    cv.file.model <- "afolu"
}else if(sector.model=="IPPU"){
    cv.file.model <- "ippu"
}else if(sector.model=="CircularEconomy"){
    cv.file.model <- "circular_economy"
}

cv.file <- sprintf("cv_results_%s_all_countries.csv", cv.file.model)

all.path.cv.file <- file.path(wd,"output_calib",sector.model,cv.file)

datos <- read.csv(all.path.cv.file)

datos$id <- as.factor(datos$id)
datos$time <- as.integer(datos$time +2011)

for (country in unique(datos$area)) {
  print(country)
  plot.file <- sprintf("%s_cv_%s_calibration.png", country,cv.file.model)
  all.path.cv.plot <- file.path(wd,"output_calib","plots",sector.model,plot.file)

  datos_sub_sim <- subset(datos, area==country & type =="simulation")
  datos_sub_hist <- subset(datos, area==country & type =="historical")
  
  
  
  ggplot(data = datos_sub_sim, aes(x=time, y=value,color = id)) +
    geom_line(alpha=0.4)+geom_line(aes(time,value ),colour="yellow",size=2, datos_sub_hist)+
    labs(title=paste0(country," Cross Validation Calibration. Run 1"),x ="Time", y = "Total CO2 emission")+ theme_minimal()
  ggsave(all.path.cv.plot , bg = "white")
  
}